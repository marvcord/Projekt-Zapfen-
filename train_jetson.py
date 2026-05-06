import argparse
import json
import math
import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, scrolledtext

try:
    from ultralytics import YOLO
    ULTRALYTICS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    YOLO = None
    ULTRALYTICS_IMPORT_ERROR = str(exc)


DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "Feder",
    1: "Fuehrungsrolle",
    2: "Sicherungsring_Fr",
    3: "Laufrolle_schmal",
    4: "Sicherungsring_Lrs",
    5: "Laufrolle_vorne",
    6: "Sicherungsring_Lrv",
    7: "Laufrolle_hinten",
    8: "Sicherungsring_Lrh",
    9: "Gussteil_gross",
    10: "Gussteil_klein",
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def normalize_path(base_dir: Path, raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def normalize_model_source(base_dir: Path, raw_value: str) -> str:
    if not raw_value:
        return ""
    if Path(raw_value).is_absolute():
        return str(Path(raw_value))
    if raw_value.startswith(".") or raw_value.startswith("~") or "/" in raw_value or "\\" in raw_value:
        return normalize_path(base_dir, raw_value)
    return raw_value


def describe_source(source: Union[int, str]) -> str:
    return f"Kamera {source}" if isinstance(source, int) else str(source)


def parse_camera_source(value: Union[int, str]) -> Union[int, str]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def bgr_to_rgb_image(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def fit_frame(frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    scale = min(max_width / width, max_height / height)
    scale = max(scale, 0.01)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def fit_frame_to_canvas(frame, canvas_width: int, canvas_height: int):
    fitted = fit_frame(frame, canvas_width, canvas_height)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    y_offset = max(0, (canvas_height - fitted.shape[0]) // 2)
    x_offset = max(0, (canvas_width - fitted.shape[1]) // 2)
    canvas[
        y_offset : y_offset + fitted.shape[0],
        x_offset : x_offset + fitted.shape[1],
    ] = fitted
    return canvas


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (149, 165, 166),
    ]
    return palette[class_id % len(palette)]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x: float
    y: float
    w: float
    h: float


@dataclass
class SideEvaluation:
    side: int
    side_name: str
    ok: bool
    status_text: str
    details: List[str] = field(default_factory=list)
    expected_count: int = 0
    detected_count: int = 0
    matches: List[Tuple[int, int, float]] = field(default_factory=list)
    wrong_positions: List[Tuple[int, int, float]] = field(default_factory=list)
    missing_indices: List[int] = field(default_factory=list)
    extra_indices: List[int] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)


@dataclass
class AppConfig:
    window_title: str = "Jetson Orin YOLOv8 Vollstaendigkeitspruefung"
    model_path: str = ""
    camera_source: Union[int, str] = 0
    camera_backend: str = "v4l2"
    camera_width: int = 1280
    camera_height: int = 720
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    match_tolerance: float = 0.09
    position_tolerance_x: float = 0.12
    position_tolerance_y: float = 0.12
    auto_start_camera: bool = True
    training_dataset_dir: str = ""
    training_base_model: str = "yolov8n.pt"
    training_epochs: int = 100
    training_imgsz: int = 640
    training_batch: int = 4
    training_device: Union[int, str] = 0
    training_workers: int = 0
    training_amp: bool = False
    class_names: Dict[int, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_NAMES))
    side_names: Dict[int, str] = field(
        default_factory=lambda: {1: "Seite 1", 2: "Seite 2", 3: "Seite 3", 4: "Seite 4"}
    )
    reference_files: Dict[int, str] = field(default_factory=dict)


def load_app_config(config_path: str) -> AppConfig:
    path = Path(config_path).expanduser().resolve()
    base_dir = path.parent
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    class_names = {
        int(key): str(value)
        for key, value in raw.get("class_names", DEFAULT_CLASS_NAMES).items()
    }
    side_names = {
        int(key): str(value)
        for key, value in raw.get(
            "side_names", {1: "Seite 1", 2: "Seite 2", 3: "Seite 3", 4: "Seite 4"}
        ).items()
    }
    reference_files = {
        int(key): normalize_path(base_dir, value)
        for key, value in raw.get("reference_files", {}).items()
        if value
    }

    model_path = raw.get("model_path", "")
    if model_path:
        model_path = normalize_path(base_dir, model_path)

    training_dataset_dir = raw.get("training_dataset_dir", "")
    if training_dataset_dir:
        training_dataset_dir = normalize_path(base_dir, training_dataset_dir)

    training_base_model = normalize_model_source(base_dir, str(raw.get("training_base_model", "yolov8n.pt")))

    return AppConfig(
        window_title=str(raw.get("window_title", AppConfig.window_title)),
        model_path=model_path,
        camera_source=parse_camera_source(raw.get("camera_source", 0)),
        camera_backend=str(raw.get("camera_backend", "v4l2")).lower(),
        camera_width=int(raw.get("camera_width", 1280)),
        camera_height=int(raw.get("camera_height", 720)),
        confidence_threshold=float(raw.get("confidence_threshold", 0.35)),
        iou_threshold=float(raw.get("iou_threshold", 0.45)),
        match_tolerance=float(raw.get("match_tolerance", 0.09)),
        position_tolerance_x=float(raw.get("position_tolerance_x", raw.get("match_tolerance", 0.12))),
        position_tolerance_y=float(raw.get("position_tolerance_y", raw.get("match_tolerance", 0.12))),
        auto_start_camera=bool(raw.get("auto_start_camera", True)),
        training_dataset_dir=training_dataset_dir,
        training_base_model=training_base_model,
        training_epochs=int(raw.get("training_epochs", 100)),
        training_imgsz=int(raw.get("training_imgsz", 640)),
        training_batch=int(raw.get("training_batch", 4)),
        training_device=parse_camera_source(raw.get("training_device", 0)),
        training_workers=0,
        training_amp=bool(raw.get("training_amp", False)),
        class_names=class_names,
        side_names=side_names,
        reference_files=reference_files,
    )


def parse_reference_file(file_path: str, class_names: Dict[int, str]) -> List[Detection]:
    entries: List[Detection] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x, y, w, h = (float(parts[index]) for index in range(1, 5))
            entries.append(
                Detection(
                    class_id=class_id,
                    class_name=class_names.get(class_id, f"Klasse_{class_id}"),
                    confidence=1.0,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                )
            )
    return entries


def normalized_distance(a: Detection, b: Detection) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def position_delta(a: Detection, b: Detection) -> Tuple[float, float, float]:
    delta_x = abs(a.x - b.x)
    delta_y = abs(a.y - b.y)
    return delta_x, delta_y, math.hypot(delta_x, delta_y)


def part_label(item: Detection) -> str:
    return f"Nr. {item.class_id} - {item.class_name}"


def relative_layout(items: List[Detection]) -> Dict[int, Tuple[float, float]]:
    if not items:
        return {}

    min_x = min(item.x - item.w / 2 for item in items)
    max_x = max(item.x + item.w / 2 for item in items)
    min_y = min(item.y - item.h / 2 for item in items)
    max_y = max(item.y + item.h / 2 for item in items)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    return {
        index: ((item.x - center_x) / span_x, (item.y - center_y) / span_y)
        for index, item in enumerate(items)
    }


def relative_delta(
    expected_layout: Dict[int, Tuple[float, float]],
    detected_layout: Dict[int, Tuple[float, float]],
    expected_index: int,
    detected_index: int,
) -> Tuple[float, float, float]:
    expected_x, expected_y = expected_layout[expected_index]
    detected_x, detected_y = detected_layout[detected_index]
    delta_x = abs(expected_x - detected_x)
    delta_y = abs(expected_y - detected_y)
    return delta_x, delta_y, math.hypot(delta_x, delta_y)


def is_within_relative_tolerance(
    expected_layout: Dict[int, Tuple[float, float]],
    detected_layout: Dict[int, Tuple[float, float]],
    expected_index: int,
    detected_index: int,
    tolerance_x: float,
    tolerance_y: float,
) -> bool:
    delta_x, delta_y, _ = relative_delta(expected_layout, detected_layout, expected_index, detected_index)
    return delta_x <= tolerance_x and delta_y <= tolerance_y


def relative_deltas_for_pairs(
    expected: List[Detection],
    detected: List[Detection],
    pairs: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    if not pairs:
        return {}

    expected_subset = [expected[expected_index] for expected_index, _ in pairs]
    detected_subset = [detected[detected_index] for _, detected_index in pairs]
    expected_layout = relative_layout(expected_subset)
    detected_layout = relative_layout(detected_subset)
    deltas: Dict[Tuple[int, int], Tuple[float, float, float]] = {}

    for pair_index, (expected_index, detected_index) in enumerate(pairs):
        delta_x, delta_y, distance = relative_delta(
            expected_layout,
            detected_layout,
            pair_index,
            pair_index,
        )
        deltas[(expected_index, detected_index)] = (delta_x, delta_y, distance)

    return deltas


def evaluate_side(
    side: int,
    side_name: str,
    expected: List[Detection],
    detected: List[Detection],
    tolerance_x: float,
    tolerance_y: float,
) -> SideEvaluation:
    expected_by_class: Dict[int, List[int]] = defaultdict(list)
    detected_by_class: Dict[int, List[int]] = defaultdict(list)

    for index, item in enumerate(expected):
        expected_by_class[item.class_id].append(index)
    for index, item in enumerate(detected):
        detected_by_class[item.class_id].append(index)

    expected_layout = relative_layout(expected)
    detected_layout = relative_layout(detected)
    preliminary_pairs: List[Tuple[int, int]] = []
    matches: List[Tuple[int, int, float]] = []
    wrong_positions: List[Tuple[int, int, float]] = []
    missing_indices: List[int] = []
    extra_indices: List[int] = []

    for class_id in sorted(set(expected_by_class) | set(detected_by_class)):
        expected_indices = expected_by_class.get(class_id, [])
        detected_indices = detected_by_class.get(class_id, [])

        primary_candidates = sorted(
            (
                relative_delta(expected_layout, detected_layout, e_idx, d_idx)[2],
                e_idx,
                d_idx,
            )
            for e_idx in expected_indices
            for d_idx in detected_indices
        )

        used_expected = set()
        used_detected = set()
        for distance, e_idx, d_idx in primary_candidates:
            if e_idx in used_expected or d_idx in used_detected:
                continue
            preliminary_pairs.append((e_idx, d_idx))
            used_expected.add(e_idx)
            used_detected.add(d_idx)

        missing_indices.extend(idx for idx in expected_indices if idx not in used_expected)
        extra_indices.extend(idx for idx in detected_indices if idx not in used_detected)

    pair_deltas = relative_deltas_for_pairs(expected, detected, preliminary_pairs)
    for e_idx, d_idx in preliminary_pairs:
        delta_x, delta_y, distance = pair_deltas.get((e_idx, d_idx), (0.0, 0.0, 0.0))
        if delta_x <= tolerance_x and delta_y <= tolerance_y:
            matches.append((e_idx, d_idx, distance))
        else:
            wrong_positions.append((e_idx, d_idx, distance))

    details: List[str] = []
    for e_idx, _, distance in matches:
        details.append(f"OK: {part_label(expected[e_idx])} erkannt (relativer Abstand {distance:.3f})")
    for e_idx, d_idx, distance in wrong_positions:
        delta_x, delta_y, _ = pair_deltas.get((e_idx, d_idx), (0.0, 0.0, distance))
        details.append(
            f"FEHLER: {part_label(expected[e_idx])} falsch montiert "
            f"(rel. dx {delta_x:.3f} > {tolerance_x:.3f} oder rel. dy {delta_y:.3f} > {tolerance_y:.3f}, Abstand {distance:.3f})"
        )
    for e_idx in missing_indices:
        details.append(f"FEHLT: {part_label(expected[e_idx])}")
    for d_idx in extra_indices:
        details.append(f"EXTRA: {part_label(detected[d_idx])}")
    if not details:
        details.append("Keine Teile in Referenz oder Detektion gefunden.")

    ok = not wrong_positions and not missing_indices and not extra_indices
    return SideEvaluation(
        side=side,
        side_name=side_name,
        ok=ok,
        status_text="I.O." if ok else "N.I.O.",
        details=details,
        expected_count=len(expected),
        detected_count=len(detected),
        matches=matches,
        wrong_positions=wrong_positions,
        missing_indices=missing_indices,
        extra_indices=extra_indices,
        detections=list(detected),
    )


def copy_file_sequentially(source_path: Path, target_path: Path, chunk_size: int = 1024 * 1024):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source_handle, target_path.open("wb") as target_handle:
        while True:
            chunk = source_handle.read(chunk_size)
            if not chunk:
                break
            target_handle.write(chunk)
    shutil.copystat(source_path, target_path)


def clean_label_text(raw_text: str) -> str:
    cleaned_lines: List[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cleaned_lines.append(" ".join(parts[:5]))
    return "\n".join(cleaned_lines)


def iter_image_files(folder_path: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ],
        key=lambda path: str(path).lower(),
    )


def ensure_unique_stem(image_dir: Path, label_dir: Path, preferred_stem: str, image_suffix: str) -> str:
    candidate = preferred_stem
    counter = 1
    while (image_dir / f"{candidate}{image_suffix}").exists() or (label_dir / f"{candidate}.txt").exists():
        candidate = f"{preferred_stem}_{counter:04d}"
        counter += 1
    return candidate


def resolve_training_source_dirs(base_dir: Path) -> Tuple[Path, Path]:
    child_dirs = {
        child.name.lower(): child
        for child in base_dir.iterdir()
        if child.is_dir()
    }

    image_candidates = ("bild", "bilder", "image", "images", "img")
    label_candidates = ("text", "texte", "txt", "label", "labels", "annotation", "annotations")

    image_dir = next((child_dirs[name] for name in image_candidates if name in child_dirs), None)
    label_dir = next((child_dirs[name] for name in label_candidates if name in child_dirs), None)

    if image_dir is not None and label_dir is not None:
        return image_dir, label_dir

    return base_dir, base_dir


class JetsonQCApp:
    def __init__(self, root: tk.Tk, config: AppConfig, config_path: Optional[str] = None):
        self.root = root
        self.config = config
        self.config_path = config_path
        self.app_dir = Path(__file__).resolve().parent
        self.session_state_path = self.app_dir / "session_state.json"
        self.weights_dir = self.app_dir / "weights"
        self.training_runs_dir = self.app_dir / "training_runs"
        self.restored_session_message = ""
        self.restore_session_state()
        self.training_dataset_dir = Path(self.config.training_dataset_dir) if self.config.training_dataset_dir else self.app_dir / "training_dataset"
        self.training_images_dir = self.training_dataset_dir / "images" / "train"
        self.training_labels_dir = self.training_dataset_dir / "labels" / "train"
        self.training_val_images_dir = self.training_dataset_dir / "images" / "val"
        self.training_val_labels_dir = self.training_dataset_dir / "labels" / "val"
        self.class_names = dict(self.config.class_names)
        self.reference_data: Dict[int, List[Detection]] = {1: [], 2: [], 3: [], 4: []}
        self.side_evaluations: Dict[int, Optional[SideEvaluation]] = {1: None, 2: None, 3: None, 4: None}
        self.current_side = 1
        self.cap = None
        self.camera_running = False
        self.model = None
        self.model_path = self.config.model_path
        self.last_frame = None
        self.preview_photo = None
        self.analysis_photo = None
        self.display_image_width = 640
        self.display_image_height = 360
        self.training_busy = False
        self.import_busy = False
        self.status_reset_after_id = None

        self.root.title(self.config.window_title)
        self.root.geometry("1600x920")
        self.root.configure(bg="#1f2933")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.ensure_training_structure()
        self.build_ui()
        self.apply_config_to_ui()

        if self.restored_session_message:
            self.append_log(self.restored_session_message)
            self.show_status(self.restored_session_message, level="info", auto_reset_ms=3500)

        if self.model_path:
            self.load_model(self.model_path, show_success=False)
        if self.config.reference_files:
            self.load_references_from_mapping(self.config.reference_files)
        if self.config.auto_start_camera:
            self.start_camera()

        self.root.after(30, self.update_video_loop)

    def build_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#1f2933")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        self.left_frame = tk.Frame(self.main_frame, bg="#1f2933")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_container = tk.Frame(self.main_frame, bg="#273947", width=540)
        self.right_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(14, 0))
        self.right_container.pack_propagate(False)

        self.right_canvas = tk.Canvas(
            self.right_container,
            bg="#273947",
            highlightthickness=0,
            bd=0,
        )
        self.right_scrollbar = tk.Scrollbar(
            self.right_container,
            orient=tk.VERTICAL,
            command=self.right_canvas.yview,
        )
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.right_canvas, bg="#273947", width=510)
        self.right_frame_window = self.right_canvas.create_window(
            (0, 0),
            window=self.right_frame,
            anchor="nw",
        )
        self.right_frame.bind("<Configure>", self.update_right_scrollregion)
        self.right_canvas.bind("<Configure>", self.update_right_canvas_width)
        self.right_canvas.bind_all("<MouseWheel>", self.handle_mousewheel)
        self.right_canvas.bind_all("<Button-4>", self.handle_mousewheel)
        self.right_canvas.bind_all("<Button-5>", self.handle_mousewheel)

        tk.Label(
            self.left_frame,
            text="Live-Bild",
            bg="#1f2933",
            fg="#f5f7fa",
            font=("Helvetica", 18, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self.video_label = tk.Label(self.left_frame, bg="#0b0f13", bd=0)
        self.video_label.pack(anchor="center")

        tk.Label(
            self.left_frame,
            text="Letzte Analyse",
            bg="#1f2933",
            fg="#d9e2ec",
            font=("Helvetica", 14, "bold"),
        ).pack(anchor="w", pady=(10, 8))

        self.analysis_label = tk.Label(self.left_frame, bg="#0b0f13", bd=0)
        self.analysis_label.pack(anchor="center")

        self.model_var = tk.StringVar(value="Modell: nicht geladen")
        self.camera_var = tk.StringVar(value="Kamera: gestoppt")
        self.dataset_var = tk.StringVar(value="Trainingsdaten: 0 Bilder")
        self.training_var = tk.StringVar(value="Training: bereit")
        self.side_var = tk.StringVar(value="Aktuelle Seite: Seite 1")
        self.overall_var = tk.StringVar(value="Gesamtergebnis: offen")
        self.confidence_control_var = tk.DoubleVar(value=self.config.confidence_threshold)
        self.tolerance_x_control_var = tk.DoubleVar(value=self.config.position_tolerance_x)
        self.tolerance_y_control_var = tk.DoubleVar(value=self.config.position_tolerance_y)

        status_frame = tk.Frame(self.right_frame, bg="#273947")
        status_frame.pack(fill=tk.X, padx=12, pady=12)

        for variable in (self.model_var, self.camera_var, self.dataset_var, self.training_var, self.side_var, self.overall_var):
            tk.Label(
                status_frame,
                textvariable=variable,
                bg="#334e68",
                fg="#f5f7fa",
                anchor="w",
                padx=10,
                pady=8,
                font=("Helvetica", 11, "bold"),
            ).pack(fill=tk.X, pady=4)

        parameter_frame = tk.LabelFrame(
            self.right_frame,
            text="Pruefparameter",
            bg="#273947",
            fg="#f5f7fa",
            padx=10,
            pady=8,
            font=("Helvetica", 11, "bold"),
        )
        parameter_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.build_parameter_control(parameter_frame, "X-Toleranz", self.tolerance_x_control_var, 0.01, 0.50, 0.01, 0)
        self.build_parameter_control(parameter_frame, "Y-Toleranz", self.tolerance_y_control_var, 0.01, 0.50, 0.01, 1)
        self.build_parameter_control(parameter_frame, "Schwellenwert", self.confidence_control_var, 0.05, 0.95, 0.05, 2)

        mode_frame = tk.Frame(parameter_frame, bg="#273947")
        mode_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        mode_frame.grid_columnconfigure(0, weight=1)
        mode_frame.grid_columnconfigure(1, weight=1)
        mode_frame.grid_columnconfigure(2, weight=1)
        mode_frame.grid_columnconfigure(3, weight=1)

        mode_specs = [
            ("Fein", "fine"),
            ("Normal", "normal"),
            ("Grob", "coarse"),
            ("Uebernehmen", "apply"),
        ]
        for column, (text, mode) in enumerate(mode_specs):
            tk.Button(
                mode_frame,
                text=text,
                command=lambda selected=mode: self.apply_parameter_mode(selected),
                bg="#486581" if mode != "apply" else "#2a9d8f",
                fg="white",
                activebackground="#486581",
                activeforeground="white",
                relief=tk.FLAT,
                padx=6,
                pady=6,
                font=("Helvetica", 9, "bold"),
                takefocus=False,
            ).grid(row=0, column=column, sticky="ew", padx=2)

        button_frame = tk.Frame(self.right_frame, bg="#273947")
        button_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.buttons: Dict[str, tk.Button] = {}
        button_specs = [
            ("camera", "Kamera Start/Stop (F1)", self.toggle_camera, "#1f7a8c"),
            ("model", "Modell laden (F2)", self.load_model_from_dialog, "#386641"),
            ("config", "Konfig laden (F3)", self.load_config_from_dialog, "#6d597a"),
            ("refs", "Referenz-TXT laden (F4)", self.load_reference_files_dialog, "#8f5a2a"),
            ("import_train", "Trainings-Hauptordner importieren (F5)", self.import_training_data_dialog, "#7a3e65"),
            ("train", "YOLO-Training starten (F6)", self.start_training_dialog, "#4c6e16"),
            ("inspect", "Aktuelle Seite pruefen (Leertaste)", self.inspect_current_side, "#2a9d8f"),
            ("reset", "Pruefzyklus resetten (R)", self.reset_cycle, "#bc6c25"),
            ("snapshot", "Snapshot speichern (S)", self.save_snapshot, "#577590"),
            ("quit", "Beenden (Q / Esc)", self.on_close, "#9d0208"),
        ]

        for index, (key, text, command, color) in enumerate(button_specs):
            button = tk.Button(
                button_frame,
                text=text,
                command=command,
                bg=color,
                fg="white",
                activebackground=color,
                activeforeground="white",
                relief=tk.FLAT,
                padx=8,
                pady=10,
                font=("Helvetica", 10, "bold"),
                wraplength=210,
                takefocus=False,
            )
            button.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=4)
            button_frame.grid_columnconfigure(index % 2, weight=1)
            self.buttons[key] = button

        side_frame = tk.LabelFrame(
            self.right_frame,
            text="Seitensteuerung",
            bg="#273947",
            fg="#f5f7fa",
            padx=10,
            pady=10,
            font=("Helvetica", 11, "bold"),
        )
        side_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        button_row = tk.Frame(side_frame, bg="#273947")
        button_row.pack(fill=tk.X)

        self.side_buttons: Dict[int, tk.Button] = {}
        self.side_status_labels: Dict[int, tk.Label] = {}
        for side in range(1, 5):
            button = tk.Button(
                button_row,
                text=str(side),
                command=lambda selected=side: self.select_side(selected),
                bg="#486581",
                fg="white",
                width=5,
                pady=6,
                font=("Helvetica", 11, "bold"),
                takefocus=False,
            )
            button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
            self.side_buttons[side] = button

        for side in range(1, 5):
            label = tk.Label(
                side_frame,
                text=f"{self.config.side_names.get(side, f'Seite {side}')}: offen",
                bg="#334e68",
                fg="#f5f7fa",
                anchor="w",
                padx=8,
                pady=4,
                font=("Helvetica", 10),
            )
            label.pack(fill=tk.X, pady=3)
            self.side_status_labels[side] = label

        help_frame = tk.LabelFrame(
            self.right_frame,
            text="Tastenfunktionen",
            bg="#273947",
            fg="#f5f7fa",
            padx=10,
            pady=10,
            font=("Helvetica", 11, "bold"),
        )
        help_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        help_text = (
            "F1  Kamera starten oder stoppen\n"
            "F2  YOLOv8 Modell laden (.pt)\n"
            "F3  JSON-Konfiguration laden\n"
            "F4  Vier Referenz-TXT laden\n"
            "F5  Trainings-Hauptordner importieren\n"
            "F6  YOLO-Training sequentiell starten\n"
            "1-4 Seite direkt waehlen\n"
            "Leertaste aktuelle Seite pruefen\n"
            "R   Pruefzyklus zuruecksetzen\n"
            "S   Snapshot speichern\n"
            "Q / Esc Programm beenden"
        )
        tk.Label(
            help_frame,
            text=help_text,
            justify=tk.LEFT,
            bg="#273947",
            fg="#d9e2ec",
            anchor="w",
            font=("Consolas", 10),
        ).pack(fill=tk.X)

        reference_frame = tk.LabelFrame(
            self.right_frame,
            text="Soll-Zustand aktuelle Seite",
            bg="#273947",
            fg="#f5f7fa",
            padx=10,
            pady=10,
            font=("Helvetica", 11, "bold"),
        )
        reference_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        self.reference_text = scrolledtext.ScrolledText(
            reference_frame,
            height=10,
            wrap=tk.WORD,
            bg="#102a43",
            fg="#f0f4f8",
            insertbackground="white",
            font=("Consolas", 10),
            takefocus=False,
        )
        self.reference_text.pack(fill=tk.BOTH, expand=True)
        self.reference_text.configure(state=tk.DISABLED)

        defect_frame = tk.LabelFrame(
            self.right_frame,
            text="Fehlerliste",
            bg="#273947",
            fg="#f5f7fa",
            padx=10,
            pady=10,
            font=("Helvetica", 11, "bold"),
        )
        defect_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        self.defect_text = scrolledtext.ScrolledText(
            defect_frame,
            height=7,
            wrap=tk.WORD,
            bg="#2b1111",
            fg="#ffe8e8",
            insertbackground="white",
            font=("Consolas", 10, "bold"),
            takefocus=False,
        )
        self.defect_text.pack(fill=tk.BOTH, expand=True)
        self.defect_text.configure(state=tk.DISABLED)

        result_frame = tk.LabelFrame(
            self.right_frame,
            text="Ergebnis / Log",
            bg="#273947",
            fg="#f5f7fa",
            padx=10,
            pady=10,
            font=("Helvetica", 11, "bold"),
        )
        result_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            height=12,
            wrap=tk.WORD,
            bg="#102a43",
            fg="#f0f4f8",
            insertbackground="white",
            font=("Consolas", 10),
            takefocus=False,
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.configure(state=tk.DISABLED)

        self.status_var = tk.StringVar(value="Bereit")
        self.status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#1f7a8c",
            fg="white",
            anchor="w",
            padx=14,
            pady=10,
            font=("Helvetica", 11, "bold"),
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=(0, 14))

        self.bind_shortcuts()
        self.root.after(200, self.restore_root_focus)
        self.show_placeholder()
        self.show_analysis_placeholder()
        self.update_defect_view()
        self.show_status("Bereit", level="info")

    def update_right_scrollregion(self, _event=None):
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def update_right_canvas_width(self, event):
        content_width = max(1, event.width)
        self.right_canvas.itemconfigure(self.right_frame_window, width=content_width)

    def handle_mousewheel(self, event):
        left = self.right_container.winfo_rootx()
        top = self.right_container.winfo_rooty()
        right = left + self.right_container.winfo_width()
        bottom = top + self.right_container.winfo_height()
        if not (left <= event.x_root <= right and top <= event.y_root <= bottom):
            return

        if event.num == 4:
            self.right_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.right_canvas.yview_scroll(1, "units")
        else:
            self.right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def build_parameter_control(
        self,
        parent: tk.Widget,
        label_text: str,
        variable: tk.DoubleVar,
        from_value: float,
        to_value: float,
        increment: float,
        row: int,
    ):
        tk.Label(
            parent,
            text=label_text,
            bg="#273947",
            fg="#d9e2ec",
            anchor="w",
            font=("Helvetica", 9, "bold"),
        ).grid(row=row, column=0, sticky="w", pady=3)

        spinbox = tk.Spinbox(
            parent,
            from_=from_value,
            to=to_value,
            increment=increment,
            textvariable=variable,
            width=7,
            format="%.2f",
            justify=tk.CENTER,
            command=self.apply_parameter_controls,
            takefocus=False,
        )
        spinbox.grid(row=row, column=1, sticky="ew", padx=6, pady=3)

        tk.Label(
            parent,
            text="relativ" if "Toleranz" in label_text else "YOLO",
            bg="#273947",
            fg="#bcccdc",
            anchor="w",
            font=("Helvetica", 9),
        ).grid(row=row, column=2, sticky="w", pady=3)

        parent.grid_columnconfigure(1, weight=1)

    def clamp_parameter(self, value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))

    def apply_parameter_mode(self, mode: str):
        if mode == "fine":
            self.tolerance_x_control_var.set(0.06)
            self.tolerance_y_control_var.set(0.06)
            self.confidence_control_var.set(0.45)
        elif mode == "normal":
            self.tolerance_x_control_var.set(0.12)
            self.tolerance_y_control_var.set(0.12)
            self.confidence_control_var.set(0.35)
        elif mode == "coarse":
            self.tolerance_x_control_var.set(0.22)
            self.tolerance_y_control_var.set(0.22)
            self.confidence_control_var.set(0.25)

        self.apply_parameter_controls()

    def apply_parameter_controls(self):
        try:
            x_tolerance = float(self.tolerance_x_control_var.get())
            y_tolerance = float(self.tolerance_y_control_var.get())
            confidence = float(self.confidence_control_var.get())
        except (tk.TclError, ValueError):
            self.show_status("Pruefparameter konnten nicht gelesen werden", level="warning", auto_reset_ms=3000)
            return

        self.config.position_tolerance_x = self.clamp_parameter(x_tolerance, 0.01, 0.50)
        self.config.position_tolerance_y = self.clamp_parameter(y_tolerance, 0.01, 0.50)
        self.config.confidence_threshold = self.clamp_parameter(confidence, 0.05, 0.95)
        self.tolerance_x_control_var.set(self.config.position_tolerance_x)
        self.tolerance_y_control_var.set(self.config.position_tolerance_y)
        self.confidence_control_var.set(self.config.confidence_threshold)
        self.update_reference_view()
        self.save_session_state()
        self.show_status("Pruefparameter uebernommen", level="success", auto_reset_ms=2500)

    def bind_shortcuts(self):
        self.root.bind_all("<F1>", self.handle_toggle_camera)
        self.root.bind_all("<F2>", self.handle_load_model)
        self.root.bind_all("<F3>", self.handle_load_config)
        self.root.bind_all("<F4>", self.handle_load_references)
        self.root.bind_all("<F5>", self.handle_import_training_data)
        self.root.bind_all("<F6>", self.handle_start_training)
        self.root.bind_all("<space>", self.handle_inspect_current_side)
        self.root.bind_all("r", self.handle_reset_cycle)
        self.root.bind_all("R", self.handle_reset_cycle)
        self.root.bind_all("s", self.handle_save_snapshot)
        self.root.bind_all("S", self.handle_save_snapshot)
        self.root.bind_all("q", self.handle_close)
        self.root.bind_all("Q", self.handle_close)
        self.root.bind_all("<Escape>", self.handle_close)
        for side in range(1, 5):
            self.root.bind_all(str(side), lambda _, selected=side: self.handle_select_side(selected))

    def restore_root_focus(self):
        try:
            self.root.focus_force()
        except tk.TclError:
            pass

    def handle_toggle_camera(self, _event=None):
        self.toggle_camera()
        return "break"

    def handle_load_model(self, _event=None):
        self.load_model_from_dialog()
        self.restore_root_focus()
        return "break"

    def handle_load_config(self, _event=None):
        self.load_config_from_dialog()
        self.restore_root_focus()
        return "break"

    def handle_load_references(self, _event=None):
        self.load_reference_files_dialog()
        self.restore_root_focus()
        return "break"

    def handle_import_training_data(self, _event=None):
        self.import_training_data_dialog()
        self.restore_root_focus()
        return "break"

    def handle_start_training(self, _event=None):
        self.start_training_dialog()
        self.restore_root_focus()
        return "break"

    def handle_inspect_current_side(self, _event=None):
        self.inspect_current_side()
        return "break"

    def handle_reset_cycle(self, _event=None):
        self.reset_cycle()
        return "break"

    def handle_save_snapshot(self, _event=None):
        self.save_snapshot()
        return "break"

    def handle_close(self, _event=None):
        self.on_close()
        return "break"

    def handle_select_side(self, side: int):
        self.select_side(side)
        return "break"

    def apply_config_to_ui(self):
        self.root.title(self.config.window_title)
        self.model_var.set(
            f"Modell: {Path(self.model_path).name}" if self.model_path else "Modell: nicht geladen"
        )
        if self.camera_running:
            self.camera_var.set(f"Kamera: aktiv ({describe_source(self.config.camera_source)})")
        else:
            self.camera_var.set(f"Kamera: bereit ({describe_source(self.config.camera_source)})")
        if hasattr(self, "confidence_control_var"):
            self.confidence_control_var.set(self.config.confidence_threshold)
            self.tolerance_x_control_var.set(self.config.position_tolerance_x)
            self.tolerance_y_control_var.set(self.config.position_tolerance_y)
        self.update_dataset_status()
        if not self.training_busy:
            self.training_var.set("Training: bereit")
        self.update_side_indicator()
        self.update_reference_view()
        self.refresh_side_status_cards()
        self.update_overall_status()

    def show_status(self, message: str, level: str = "info", auto_reset_ms: Optional[int] = None):
        colors = {
            "info": ("#1f7a8c", "white"),
            "success": ("#2f855a", "white"),
            "warning": ("#bc6c25", "white"),
            "error": ("#c53030", "white"),
            "busy": ("#6d597a", "white"),
        }
        bg_color, fg_color = colors.get(level, colors["info"])
        self.status_var.set(message)
        self.status_bar.configure(bg=bg_color, fg=fg_color)

        if self.status_reset_after_id is not None:
            self.root.after_cancel(self.status_reset_after_id)
            self.status_reset_after_id = None

        if auto_reset_ms is not None:
            self.status_reset_after_id = self.root.after(
                auto_reset_ms,
                lambda: self.show_status("Bereit", level="info"),
            )

    def show_status_from_worker(self, message: str, level: str = "info", auto_reset_ms: Optional[int] = None):
        self.root.after(0, lambda: self.show_status(message, level=level, auto_reset_ms=auto_reset_ms))

    def restore_session_state(self):
        if not self.session_state_path.exists():
            return

        try:
            raw_state = json.loads(self.session_state_path.read_text(encoding="utf-8"))
        except Exception:
            self.restored_session_message = "Gespeicherte Sitzung konnte nicht gelesen werden."
            return

        self.config_path = raw_state.get("config_path") or self.config_path
        self.config.window_title = str(raw_state.get("window_title", self.config.window_title))
        self.config.model_path = str(raw_state.get("model_path", self.config.model_path or ""))
        self.config.camera_source = parse_camera_source(raw_state.get("camera_source", self.config.camera_source))
        self.config.camera_backend = str(raw_state.get("camera_backend", self.config.camera_backend))
        self.config.camera_width = int(raw_state.get("camera_width", self.config.camera_width))
        self.config.camera_height = int(raw_state.get("camera_height", self.config.camera_height))
        self.config.confidence_threshold = float(raw_state.get("confidence_threshold", self.config.confidence_threshold))
        self.config.iou_threshold = float(raw_state.get("iou_threshold", self.config.iou_threshold))
        self.config.match_tolerance = float(raw_state.get("match_tolerance", self.config.match_tolerance))
        self.config.position_tolerance_x = float(raw_state.get("position_tolerance_x", self.config.position_tolerance_x))
        self.config.position_tolerance_y = float(raw_state.get("position_tolerance_y", self.config.position_tolerance_y))
        self.config.auto_start_camera = bool(raw_state.get("auto_start_camera", self.config.auto_start_camera))
        self.config.training_dataset_dir = str(raw_state.get("training_dataset_dir", self.config.training_dataset_dir or ""))
        self.config.training_base_model = str(raw_state.get("training_base_model", self.config.training_base_model))
        self.config.training_epochs = int(raw_state.get("training_epochs", self.config.training_epochs))
        self.config.training_imgsz = int(raw_state.get("training_imgsz", self.config.training_imgsz))
        self.config.training_batch = int(raw_state.get("training_batch", self.config.training_batch))
        self.config.training_device = parse_camera_source(raw_state.get("training_device", self.config.training_device))
        self.config.training_workers = 0
        self.config.training_amp = bool(raw_state.get("training_amp", self.config.training_amp))
        self.config.class_names = {
            int(key): str(value)
            for key, value in raw_state.get("class_names", self.config.class_names).items()
        }
        self.config.side_names = {
            int(key): str(value)
            for key, value in raw_state.get("side_names", self.config.side_names).items()
        }
        self.config.reference_files = {
            int(key): str(value)
            for key, value in raw_state.get("reference_files", {}).items()
            if value
        }

        self.model_path = self.config.model_path
        self.restored_session_message = "Letzte Sitzung wurde automatisch wiederhergestellt."

    def build_session_state(self) -> Dict[str, object]:
        return {
            "config_path": self.config_path or "",
            "window_title": self.config.window_title,
            "model_path": self.model_path or "",
            "camera_source": self.config.camera_source,
            "camera_backend": self.config.camera_backend,
            "camera_width": self.config.camera_width,
            "camera_height": self.config.camera_height,
            "confidence_threshold": self.config.confidence_threshold,
            "iou_threshold": self.config.iou_threshold,
            "match_tolerance": self.config.match_tolerance,
            "position_tolerance_x": self.config.position_tolerance_x,
            "position_tolerance_y": self.config.position_tolerance_y,
            "auto_start_camera": self.config.auto_start_camera,
            "training_dataset_dir": str(self.training_dataset_dir),
            "training_base_model": self.config.training_base_model,
            "training_epochs": self.config.training_epochs,
            "training_imgsz": self.config.training_imgsz,
            "training_batch": self.config.training_batch,
            "training_device": self.config.training_device,
            "training_workers": 0,
            "training_amp": self.config.training_amp,
            "class_names": {str(key): value for key, value in self.class_names.items()},
            "side_names": {str(key): value for key, value in self.config.side_names.items()},
            "reference_files": {
                str(key): value
                for key, value in self.config.reference_files.items()
                if value
            },
        }

    def save_session_state(self, log_errors: bool = False):
        try:
            self.session_state_path.write_text(
                json.dumps(self.build_session_state(), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            if log_errors and hasattr(self, "result_text"):
                self.append_log(f"Sitzung konnte nicht gespeichert werden: {exc}")

    def ensure_training_structure(self):
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.training_images_dir.mkdir(parents=True, exist_ok=True)
        self.training_labels_dir.mkdir(parents=True, exist_ok=True)
        self.training_val_images_dir.mkdir(parents=True, exist_ok=True)
        self.training_val_labels_dir.mkdir(parents=True, exist_ok=True)
        self.training_runs_dir.mkdir(parents=True, exist_ok=True)

    def count_images_in_dir(self, directory: Path) -> int:
        if not directory.exists():
            return 0
        return sum(
            1
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )

    def count_training_images(self) -> int:
        return self.count_images_in_dir(self.training_images_dir) + self.count_images_in_dir(self.training_val_images_dir)

    def clear_training_dataset_cache(self):
        for cache_path in self.training_dataset_dir.rglob('*.cache'):
            try:
                cache_path.unlink()
            except OSError:
                pass

    def update_dataset_status(self):
        image_count = self.count_training_images()
        train_count = self.count_images_in_dir(self.training_images_dir)
        val_count = self.count_images_in_dir(self.training_val_images_dir)
        self.dataset_var.set(f"Trainingsdaten: {image_count} Bilder (Train: {train_count}, Val: {val_count})")

    def append_log_from_worker(self, message: str):
        self.root.after(0, lambda: self.append_log(message))

    def set_training_state(self, message: str):
        self.root.after(0, lambda: self.training_var.set(message))

    def set_background_job_state(self, *, import_busy: Optional[bool] = None, training_busy: Optional[bool] = None):
        if import_busy is not None:
            self.import_busy = import_busy
        if training_busy is not None:
            self.training_busy = training_busy

        busy = self.import_busy or self.training_busy
        self.buttons["import_train"].configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.buttons["train"].configure(state=tk.DISABLED if busy else tk.NORMAL)

    def append_log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.insert(tk.END, line + "\n")
        self.result_text.see(tk.END)
        self.result_text.configure(state=tk.DISABLED)

    def build_solid_frame(self, width: int, height: int, color: Tuple[int, int, int]):
        blue, green, red = color
        return cv2.merge(
            [
                blue * np.ones((height, width), dtype="uint8"),
                green * np.ones((height, width), dtype="uint8"),
                red * np.ones((height, width), dtype="uint8"),
            ]
        )

    def build_placeholder_frame(self, width: int, height: int, title: str, subtitle: str):
        frame = self.build_solid_frame(width, height, (11, 15, 19))
        cv2.putText(frame, title, (60, height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 244, 248), 2)
        cv2.putText(frame, subtitle, (60, height // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 190, 199), 2)
        return frame

    def show_frame_on_label(self, label: tk.Label, frame, target: str):
        fitted = fit_frame_to_canvas(frame, self.display_image_width, self.display_image_height)
        image = ImageTk.PhotoImage(bgr_to_rgb_image(fitted))
        if target == "preview":
            self.preview_photo = image
        else:
            self.analysis_photo = image
        label.configure(image=image)

    def show_placeholder(self):
        frame = self.build_placeholder_frame(
            width=1280,
            height=720,
            title="Keine Kamera aktiv",
            subtitle="F1 oder Button 'Kamera Start/Stop' zum Starten",
        )
        self.show_frame_on_label(self.video_label, frame, "preview")

    def show_analysis_placeholder(self):
        frame = self.build_placeholder_frame(
            width=1280,
            height=360,
            title="Noch keine Analyse",
            subtitle="Leertaste oder Button 'Aktuelle Seite pruefen'",
        )
        self.show_frame_on_label(self.analysis_label, frame, "analysis")

    def start_camera(self):
        if self.camera_running:
            return

        source = self.config.camera_source
        backend_name = self.config.camera_backend.lower()
        backend = cv2.CAP_ANY
        if backend_name == "v4l2":
            backend = cv2.CAP_V4L2
        elif backend_name == "gstreamer":
            backend = cv2.CAP_GSTREAMER
        elif backend_name == "any":
            backend = cv2.CAP_ANY

        if isinstance(source, str) and ("!" in source or "nvarguscamerasrc" in source):
            backend = cv2.CAP_GSTREAMER

        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap or not self.cap.isOpened():
            self.camera_running = False
            self.camera_var.set(f"Kamera: Fehler bei {describe_source(source)}")
            self.append_log(f"Kamera konnte nicht geoeffnet werden: {describe_source(source)}")
            self.show_status(f"Kamera konnte nicht geoeffnet werden: {describe_source(source)}", level="error")
            self.show_placeholder()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        self.camera_running = True
        self.camera_var.set(f"Kamera: aktiv ({describe_source(source)})")
        self.append_log(f"Kamera gestartet: {describe_source(source)}")
        self.show_status(f"Kamera aktiv: {describe_source(source)}", level="success", auto_reset_ms=2500)

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_running = False
        self.last_frame = None
        self.camera_var.set(f"Kamera: gestoppt ({describe_source(self.config.camera_source)})")
        self.append_log("Kamera gestoppt.")
        self.show_status("Kamera gestoppt", level="warning", auto_reset_ms=2500)
        self.show_placeholder()

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def draw_preview_overlay(self, frame):
        side_name = self.config.side_names.get(self.current_side, f"Seite {self.current_side}")
        cv2.rectangle(frame, (20, 20), (430, 115), (20, 28, 36), -1)
        cv2.putText(frame, "Jetson QC - Live", (36, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 244, 248), 2)
        cv2.putText(frame, f"Aktuelle Seite: {side_name}", (36, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (217, 226, 236), 2)
        cv2.putText(frame, "Leertaste prueft, R reset, S Snapshot", (36, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 190, 199), 1)
        return frame

    def update_video_loop(self):
        if self.camera_running and self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                self.last_frame = frame.copy()
                self.show_frame_on_label(self.video_label, self.draw_preview_overlay(frame.copy()), "preview")
            else:
                self.append_log("Warnung: Frame konnte nicht gelesen werden.")
        self.root.after(30, self.update_video_loop)

    def load_model(self, model_path: str, show_success: bool = True):
        if YOLO is None:
            messagebox.showerror(
                "Ultralytics fehlt",
                "Das Paket 'ultralytics' konnte nicht importiert werden.\n"
                f"Fehler: {ULTRALYTICS_IMPORT_ERROR}",
            )
            self.append_log(f"YOLOv8 Import fehlgeschlagen: {ULTRALYTICS_IMPORT_ERROR}")
            return False

        if not os.path.exists(model_path):
            self.append_log(f"Modelldatei nicht gefunden: {model_path}")
            self.show_status("Modelldatei nicht gefunden", level="error")
            if show_success:
                messagebox.showerror("Modell fehlt", f"Datei nicht gefunden:\n{model_path}")
            return False

        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.model_var.set(f"Modell: {Path(model_path).name}")
            self.append_log(f"YOLOv8 Modell geladen: {model_path}")
            self.show_status(f"Modell geladen: {Path(model_path).name}", level="success", auto_reset_ms=3000)
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                self.class_names = {int(key): str(value) for key, value in self.model.names.items()}
                self.config.class_names = dict(self.class_names)
            self.update_reference_view()
            self.save_session_state()
            if show_success:
                messagebox.showinfo("Modell geladen", f"Modell geladen:\n{model_path}")
            return True
        except Exception as exc:
            self.append_log(f"Modell konnte nicht geladen werden: {exc}")
            self.show_status("Modell konnte nicht geladen werden", level="error")
            messagebox.showerror("Modellfehler", str(exc))
            return False

    def load_model_from_dialog(self):
        selected = filedialog.askopenfilename(
            title="YOLOv8 Modell waehlen",
            filetypes=[("PyTorch Modelle", "*.pt"), ("Alle Dateien", "*.*")],
        )
        if selected:
            self.load_model(selected, show_success=True)

    def load_reference_files_dialog(self):
        selected_files: Dict[int, str] = {}
        for side in range(1, 5):
            selected = filedialog.askopenfilename(
                title=f"Referenz-TXT fuer {self.config.side_names.get(side, f'Seite {side}')} waehlen",
                filetypes=[("Textdateien", "*.txt"), ("Alle Dateien", "*.*")],
            )
            if not selected:
                self.append_log("Referenzladen abgebrochen.")
                return
            selected_files[side] = selected
        self.load_references_from_mapping(selected_files)

    def load_references_from_mapping(self, mapping: Dict[int, str]):
        loaded_count = 0
        missing_files: List[str] = []
        failed_files: List[str] = []

        for side in range(1, 5):
            file_path = mapping.get(side)
            if not file_path:
                continue

            side_name = self.config.side_names.get(side, f"Seite {side}")
            if not os.path.exists(file_path):
                missing_files.append(f"{side_name}: {file_path}")
                self.append_log(f"Referenzdatei fehlt fuer {side_name}: {file_path}")
                continue

            try:
                entries = parse_reference_file(file_path, self.class_names)
                self.reference_data[side] = entries
                self.config.reference_files[side] = file_path
                loaded_count += 1
                self.append_log(f"Referenz fuer {side_name} geladen: {len(entries)} Soll-Teile")
            except Exception as exc:
                failed_files.append(f"{side_name}: {file_path} ({exc})")
                self.append_log(f"Fehler beim Laden der Referenz fuer {side_name}: {exc}")

        self.update_reference_view()
        self.refresh_side_status_cards()

        if loaded_count > 0:
            self.show_status(
                f"Referenzen geladen: {loaded_count} von 4",
                level="success" if not missing_files and not failed_files else "warning",
                auto_reset_ms=3500,
            )
        elif missing_files or failed_files:
            self.show_status("Konfiguration geladen, aber Referenzen fehlen", level="warning", auto_reset_ms=4500)

        if failed_files:
            messagebox.showwarning(
                "Referenzhinweis",
                "Die Konfiguration wurde geladen, aber einige Referenzen konnten nicht gelesen werden.\n\n"
                + "\n".join(failed_files[:4]),
            )

        self.save_session_state()
        return loaded_count > 0

    def load_config_from_dialog(self):
        selected = filedialog.askopenfilename(
            title="JSON-Konfiguration waehlen",
            filetypes=[("JSON Dateien", "*.json"), ("Alle Dateien", "*.*")],
        )
        if selected:
            self.load_config(selected)

    def load_config(self, config_path: str):
        try:
            new_config = load_app_config(config_path)
        except Exception as exc:
            self.append_log(f"Konfiguration konnte nicht geladen werden: {exc}")
            messagebox.showerror("Konfigurationsfehler", str(exc))
            return False

        self.config = new_config
        self.config_path = config_path
        self.training_dataset_dir = Path(self.config.training_dataset_dir) if self.config.training_dataset_dir else self.app_dir / "training_dataset"
        self.training_images_dir = self.training_dataset_dir / "images" / "train"
        self.training_labels_dir = self.training_dataset_dir / "labels" / "train"
        self.training_val_images_dir = self.training_dataset_dir / "images" / "val"
        self.training_val_labels_dir = self.training_dataset_dir / "labels" / "val"
        self.class_names = dict(new_config.class_names)
        self.reference_data = {1: [], 2: [], 3: [], 4: []}
        self.side_evaluations = {1: None, 2: None, 3: None, 4: None}
        self.current_side = 1

        if self.camera_running:
            self.stop_camera()

        self.model = None
        self.model_path = new_config.model_path
        self.ensure_training_structure()
        self.apply_config_to_ui()
        self.append_log(f"Konfiguration geladen: {config_path}")
        self.save_session_state()

        if self.model_path:
            self.load_model(self.model_path, show_success=False)
        if self.config.reference_files:
            self.load_references_from_mapping(self.config.reference_files)
        if self.config.auto_start_camera:
            self.start_camera()
        return True

    def import_training_data_dialog(self):
        if self.import_busy or self.training_busy:
            messagebox.showinfo("Bitte warten", "Es laeuft bereits ein Import oder Training.")
            return

        base_dir = filedialog.askdirectory(title="Trainings-Hauptordner waehlen")
        if not base_dir:
            return

        base_path = Path(base_dir)
        image_dir, label_dir = resolve_training_source_dirs(base_path)

        if image_dir == base_path and label_dir == base_path:
            self.append_log(
                f"Keine separaten Unterordner 'Bild'/'Text' gefunden. Nutze den gesamten Ordner: {base_path}"
            )
        else:
            self.append_log(
                f"Trainingsquellen erkannt: Bilder aus {image_dir.name}, Labels aus {label_dir.name}"
            )

        self.set_background_job_state(import_busy=True)
        self.training_var.set("Training: Import laeuft")
        self.append_log(f"Trainingsimport gestartet: Hauptordner {base_path}")
        self.show_status("Trainingsimport laeuft...", level="busy")
        threading.Thread(
            target=self.import_training_data_worker,
            args=(Path(image_dir), Path(label_dir)),
            daemon=True,
        ).start()

    def import_training_data_worker(self, image_dir: Path, label_dir: Path):
        try:
            image_files = iter_image_files(image_dir)
            label_files = {
                path.stem: path
                for path in sorted(label_dir.rglob("*.txt"), key=lambda path: str(path).lower())
                if path.is_file()
            }

            total_files = len(image_files)
            if total_files == 0:
                self.append_log_from_worker("Keine Bilddateien fuer den Trainingsimport gefunden.")
                self.show_status_from_worker("Keine Bilddateien gefunden", level="warning", auto_reset_ms=4000)
                return
            if not label_files:
                self.append_log_from_worker("Keine TXT-Labeldateien fuer den Trainingsimport gefunden.")
                self.show_status_from_worker("Keine TXT-Labeldateien gefunden", level="warning", auto_reset_ms=4000)
                return

            paired_files = []
            skipped_count = 0
            for image_path in image_files:
                label_path = label_files.get(image_path.stem)
                if label_path is None:
                    skipped_count += 1
                    self.append_log_from_worker(f"Uebersprungen ohne Label: {image_path.name}")
                    continue
                paired_files.append((image_path, label_path))

            if not paired_files:
                self.append_log_from_worker("Keine gueltigen Bild-Label-Paare fuer den Trainingsimport gefunden.")
                self.show_status_from_worker("Keine gueltigen Bild-Label-Paare gefunden", level="warning", auto_reset_ms=4000)
                return

            imported_count = 0
            val_target_count = 0
            if len(paired_files) > 1:
                val_target_count = max(1, round(len(paired_files) * 0.2))

            val_indices = set(range(len(paired_files) - val_target_count, len(paired_files))) if val_target_count else set()

            for index, (image_path, label_path) in enumerate(paired_files, start=1):
                dataset_index = index - 1
                if dataset_index in val_indices:
                    target_image_dir = self.training_val_images_dir
                    target_label_dir = self.training_val_labels_dir
                else:
                    target_image_dir = self.training_images_dir
                    target_label_dir = self.training_labels_dir

                unique_stem = ensure_unique_stem(
                    target_image_dir,
                    target_label_dir,
                    image_path.stem,
                    image_path.suffix.lower(),
                )
                target_image = target_image_dir / f"{unique_stem}{image_path.suffix.lower()}"
                target_label = target_label_dir / f"{unique_stem}.txt"

                copy_file_sequentially(image_path, target_image)
                raw_label_text = label_path.read_text(encoding="utf-8", errors="ignore")
                target_label.write_text(clean_label_text(raw_label_text), encoding="utf-8")
                imported_count += 1

                if index % 10 == 0 or index == len(paired_files):
                    self.append_log_from_worker(
                        f"Import Fortschritt: {index}/{len(paired_files)} Paare verarbeitet | "
                        f"{imported_count} importiert | {skipped_count} uebersprungen"
                    )
                    self.root.after(0, self.update_dataset_status)

            self.clear_training_dataset_cache()
            yaml_path = self.write_training_yaml()
            train_count = self.count_images_in_dir(self.training_images_dir)
            val_count = self.count_images_in_dir(self.training_val_images_dir)
            self.append_log_from_worker(
                f"Trainingsimport abgeschlossen: {imported_count} Paare importiert, {skipped_count} uebersprungen"
            )
            self.append_log_from_worker(
                f"Datensatz aufgeteilt: Train={train_count}, Val={val_count}"
            )
            self.append_log_from_worker(f"Dataset-Konfiguration aktualisiert: {yaml_path}")
            self.show_status_from_worker(
                f"Import fertig: {imported_count} Paare importiert",
                level="success",
                auto_reset_ms=4000,
            )
        except Exception as exc:
            self.append_log_from_worker(f"Fehler beim Trainingsimport: {exc}")
            self.show_status_from_worker("Fehler beim Trainingsimport", level="error", auto_reset_ms=5000)
        finally:
            self.root.after(0, self.update_dataset_status)
            self.root.after(0, lambda: self.training_var.set("Training: bereit"))
            self.root.after(0, lambda: self.set_background_job_state(import_busy=False))

    def write_training_yaml(self) -> Path:
        self.ensure_training_structure()
        yaml_path = self.training_dataset_dir / "dataset.yaml"
        class_items = sorted(self.class_names.items())
        lines = [
            f"path: {self.training_dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(class_items)}",
            "",
            "names:",
        ]
        for class_id, class_name in class_items:
            lines.append(f"  {class_id}: {class_name}")
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path

    def start_training_dialog(self):
        if self.import_busy or self.training_busy:
            messagebox.showinfo("Bitte warten", "Es laeuft bereits ein Import oder Training.")
            return
        if YOLO is None:
            messagebox.showerror(
                "Ultralytics fehlt",
                "Das Paket 'ultralytics' konnte nicht importiert werden.\n"
                f"Fehler: {ULTRALYTICS_IMPORT_ERROR}",
            )
            return

        dataset_size = self.count_training_images()
        if dataset_size == 0:
            messagebox.showwarning("Keine Trainingsdaten", "Bitte zuerst Trainingsdaten importieren.")
            return

        if not messagebox.askyesno(
            "Training starten",
            "Soll das YOLO-Training jetzt gestartet werden?\n"
            "Die Daten werden dabei absichtlich sequentiell mit workers=0 geladen.\n"
            "Fuer den Jetson wird AMP standardmaessig deaktiviert.",
        ):
            return

        self.set_background_job_state(training_busy=True)
        self.training_var.set("Training: laeuft sequentiell")
        self.append_log(f"YOLO-Training gestartet mit {dataset_size} Bildern | workers=0")
        self.show_status("YOLO-Training laeuft...", level="busy")
        threading.Thread(target=self.training_worker, daemon=True).start()

    def training_worker(self):
        try:
            self.clear_training_dataset_cache()
            yaml_path = self.write_training_yaml()
            sequential_workers = 0
            if self.model_path and Path(self.model_path).exists():
                model_source = self.model_path
            else:
                model_source = self.config.training_base_model

            self.append_log_from_worker(f"Trainingsmodell: {model_source}")
            self.append_log_from_worker(
                f"Datensatz wird sequentiell gelesen (workers={sequential_workers}, cache=False)."
            )
            self.append_log_from_worker(
                f"Jetson-Trainingsmodus: amp={self.config.training_amp}, batch={self.config.training_batch}."
            )

            training_model = YOLO(model_source)
            results = training_model.train(
                data=str(yaml_path),
                epochs=self.config.training_epochs,
                imgsz=self.config.training_imgsz,
                batch=self.config.training_batch,
                device=self.config.training_device,
                workers=sequential_workers,
                amp=self.config.training_amp,
                cache=False,
                project=str(self.training_runs_dir),
                name="sequential_training",
                exist_ok=False,
                verbose=False,
            )

            best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            if best_model_path.exists():
                target_best_model = self.weights_dir / "best.pt"
                copy_file_sequentially(best_model_path, target_best_model)
                self.model_path = str(target_best_model)
                self.root.after(0, lambda: self.load_model(str(target_best_model), show_success=False))
                self.append_log_from_worker(f"Neues Modell gespeichert: {target_best_model}")

            self.append_log_from_worker(f"Training abgeschlossen. Ergebnisse unter: {results.save_dir}")
            self.show_status_from_worker("YOLO-Training abgeschlossen", level="success", auto_reset_ms=5000)
        except Exception as exc:
            self.append_log_from_worker(f"Fehler beim YOLO-Training: {exc}")
            self.show_status_from_worker("Fehler beim YOLO-Training", level="error", auto_reset_ms=5000)
        finally:
            self.root.after(0, lambda: self.training_var.set("Training: bereit"))
            self.root.after(0, lambda: self.set_background_job_state(training_busy=False))

    def select_side(self, side: int):
        self.current_side = side
        self.update_side_indicator()
        self.update_reference_view()
        evaluation = self.side_evaluations.get(side)
        if evaluation is None:
            self.update_defect_view()
        else:
            self.update_defect_view(evaluation, self.reference_data.get(side, []))

    def update_side_indicator(self):
        side_name = self.config.side_names.get(self.current_side, f"Seite {self.current_side}")
        self.side_var.set(f"Aktuelle Seite: {side_name}")
        for side, button in self.side_buttons.items():
            button.configure(bg="#d97706" if side == self.current_side else "#486581")

    def update_reference_view(self):
        side_name = self.config.side_names.get(self.current_side, f"Seite {self.current_side}")
        entries = self.reference_data.get(self.current_side, [])
        counts = Counter(item.class_name for item in entries)

        lines = [side_name, ""]
        if not entries:
            lines.append("Noch keine Referenz fuer diese Seite geladen.")
            lines.append("Erwarte 4 YOLO-Labeldateien, eine pro Seite.")
        else:
            lines.append(f"Soll-Teile gesamt: {len(entries)}")
            lines.append(
                f"Positions-Toleranz X/Y: {self.config.position_tolerance_x:.3f} / {self.config.position_tolerance_y:.3f}"
            )
            lines.append("")
            lines.append("Klassen / Anzahl:")
            for class_name, amount in sorted(counts.items()):
                lines.append(f"- {class_name}: {amount}")

        self.reference_text.configure(state=tk.NORMAL)
        self.reference_text.delete("1.0", tk.END)
        self.reference_text.insert(tk.END, "\n".join(lines))
        self.reference_text.configure(state=tk.DISABLED)

    def update_defect_view(
        self,
        evaluation: Optional[SideEvaluation] = None,
        expected: Optional[List[Detection]] = None,
    ):
        lines: List[str] = []
        if evaluation is None or expected is None:
            lines.append("Noch keine Auswertung vorhanden.")
        elif evaluation.ok:
            lines.append("Keine fehlenden oder falsch montierten Teile.")
        else:
            paired_items = [
                (expected_index, detected_index)
                for expected_index, detected_index, _ in evaluation.matches + evaluation.wrong_positions
            ]
            pair_deltas = relative_deltas_for_pairs(expected, evaluation.detections, paired_items)

            for expected_index, detected_index, _ in evaluation.wrong_positions:
                expected_item = expected[expected_index]
                detected_item = evaluation.detections[detected_index]
                delta_x, delta_y, distance = pair_deltas.get((expected_index, detected_index), (0.0, 0.0, 0.0))
                lines.append(
                    f"FALSCH MONTIERT: {part_label(expected_item)} "
                    f"(rel. dx={delta_x:.3f}, dy={delta_y:.3f}, Abstand={distance:.3f})"
                )
                lines.append(f"Erkannt als: {part_label(detected_item)}")

            for expected_index in evaluation.missing_indices:
                lines.append(f"FEHLT: {part_label(expected[expected_index])}")

            for detected_index in evaluation.extra_indices:
                lines.append(f"ZUSAETZLICH ERKANNT: {part_label(evaluation.detections[detected_index])}")

        self.defect_text.configure(state=tk.NORMAL)
        self.defect_text.delete("1.0", tk.END)
        self.defect_text.insert(tk.END, "\n".join(lines))
        self.defect_text.configure(state=tk.DISABLED)

    def refresh_side_status_cards(self):
        for side in range(1, 5):
            label = self.side_status_labels[side]
            evaluation = self.side_evaluations.get(side)
            side_name = self.config.side_names.get(side, f"Seite {side}")
            reference_loaded = len(self.reference_data.get(side, [])) > 0

            if evaluation is None:
                text = f"{side_name}: offen"
                color = "#334e68" if reference_loaded else "#7c4d00"
            elif evaluation.ok:
                text = f"{side_name}: I.O. ({evaluation.expected_count}/{evaluation.detected_count})"
                color = "#2f855a"
            else:
                text = f"{side_name}: N.I.O. ({evaluation.expected_count}/{evaluation.detected_count})"
                color = "#c53030"
            label.configure(text=text, bg=color)

    def update_overall_status(self):
        completed = [evaluation for evaluation in self.side_evaluations.values() if evaluation is not None]
        if len(completed) < 4:
            self.overall_var.set(f"Gesamtergebnis: {len(completed)}/4 Seiten geprueft")
        elif all(evaluation.ok for evaluation in completed):
            self.overall_var.set("Gesamtergebnis: I.O. - Bauteil vollstaendig")
        else:
            self.overall_var.set("Gesamtergebnis: N.I.O. - Fehler gefunden")

    def run_detection(self, frame) -> List[Detection]:
        detections: List[Detection] = []
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )

        image_height, image_width = frame.shape[:2]
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x_center = ((x1 + x2) / 2.0) / image_width
                y_center = ((y1 + y2) / 2.0) / image_height
                width = (x2 - x1) / image_width
                height = (y2 - y1) / image_height
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self.class_names.get(class_id, f"Klasse_{class_id}"),
                        confidence=confidence,
                        x=x_center,
                        y=y_center,
                        w=width,
                        h=height,
                    )
                )
        return detections

    def draw_analysis_overlay(self, frame, expected: List[Detection], evaluation: SideEvaluation):
        height, width = frame.shape[:2]
        matched_expected = {e_idx for e_idx, _, _ in evaluation.matches}
        matched_detected = {d_idx for _, d_idx, _ in evaluation.matches}
        wrong_expected = {e_idx for e_idx, _, _ in evaluation.wrong_positions}
        wrong_detected = {d_idx for _, d_idx, _ in evaluation.wrong_positions}

        for expected_index, item in enumerate(expected):
            x1 = int((item.x - item.w / 2) * width)
            y1 = int((item.y - item.h / 2) * height)
            x2 = int((item.x + item.w / 2) * width)
            y2 = int((item.y + item.h / 2) * height)
            if expected_index in matched_expected:
                color = (46, 204, 113)
            elif expected_index in wrong_expected:
                color = (0, 196, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"SOLL {item.class_name}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for detected_index, item in enumerate(evaluation.detections):
            x1 = int((item.x - item.w / 2) * width)
            y1 = int((item.y - item.h / 2) * height)
            x2 = int((item.x + item.w / 2) * width)
            y2 = int((item.y + item.h / 2) * height)
            if detected_index in matched_detected:
                color = color_for_class(item.class_id)
            elif detected_index in wrong_detected:
                color = (0, 196, 255)
            elif detected_index in evaluation.extra_indices:
                color = (255, 0, 255)
            else:
                color = (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"IST {item.class_name} {item.confidence:.2f}", (x1, min(height - 10, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        header_color = (46, 204, 113) if evaluation.ok else (231, 76, 60)
        cv2.rectangle(frame, (18, 18), (640, 118), (20, 28, 36), -1)
        cv2.putText(frame, f"{evaluation.side_name}: {evaluation.status_text}", (34, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.95, header_color, 2)
        cv2.putText(
            frame,
            f"Soll: {evaluation.expected_count} | Ist: {evaluation.detected_count} | Tol X/Y: {self.config.position_tolerance_x:.3f}/{self.config.position_tolerance_y:.3f}",
            (34, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (217, 226, 236),
            2,
        )
        cv2.putText(frame, "Gruen=korrekt  Cyan=falsche Position  Rot=fehlt  Magenta=extra", (34, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 190, 199), 1)
        return frame

    def inspect_current_side(self):
        self.append_log(f"Pruefung ausgelost fuer {self.config.side_names.get(self.current_side, f'Seite {self.current_side}')}.")
        self.show_status(
            f"Pruefung gestartet: {self.config.side_names.get(self.current_side, f'Seite {self.current_side}')}",
            level="busy",
        )
        if not self.camera_running or self.last_frame is None:
            self.append_log("Pruefung abgebrochen: kein aktuelles Kamerabild vorhanden.")
            self.show_status("Pruefung nicht moeglich: kein Kamerabild", level="warning", auto_reset_ms=3500)
            messagebox.showwarning("Keine Kamera", "Bitte zuerst die Kamera starten.")
            return
        if self.model is None:
            self.append_log("Pruefung abgebrochen: kein YOLO-Modell geladen.")
            self.show_status("Pruefung nicht moeglich: kein YOLO-Modell", level="warning", auto_reset_ms=3500)
            messagebox.showwarning("Kein Modell", "Bitte zuerst ein YOLOv8 Modell laden.")
            return

        expected = self.reference_data.get(self.current_side, [])
        if not expected:
            self.append_log("Pruefung abgebrochen: keine Referenz fuer die aktuelle Seite geladen.")
            self.show_status("Pruefung nicht moeglich: keine Referenz geladen", level="warning", auto_reset_ms=3500)
            messagebox.showwarning("Keine Referenz", f"Bitte zuerst die Referenzdatei fuer {self.config.side_names.get(self.current_side, f'Seite {self.current_side}')} laden.")
            return

        frame = self.last_frame.copy()
        detections = self.run_detection(frame)
        side_name = self.config.side_names.get(self.current_side, f"Seite {self.current_side}")
        evaluation = evaluate_side(
            side=self.current_side,
            side_name=side_name,
            expected=expected,
            detected=detections,
            tolerance_x=self.config.position_tolerance_x,
            tolerance_y=self.config.position_tolerance_y,
        )
        self.side_evaluations[self.current_side] = evaluation
        self.refresh_side_status_cards()
        self.update_overall_status()

        self.append_log(f"{side_name} geprueft: {evaluation.status_text} | Soll {evaluation.expected_count} | Ist {evaluation.detected_count}")
        for detail in evaluation.details:
            self.append_log(f"  {detail}")
        self.update_defect_view(evaluation, expected)
        if evaluation.ok:
            self.show_status(f"{side_name}: I.O.", level="success", auto_reset_ms=3500)
        else:
            self.show_status(f"{side_name}: N.I.O.", level="error", auto_reset_ms=4500)

        overlay = self.draw_analysis_overlay(frame, expected, evaluation)
        self.show_frame_on_label(self.analysis_label, overlay, "analysis")
        if evaluation.ok and self.current_side < 4:
            self.select_side(self.current_side + 1)

    def save_snapshot(self):
        self.append_log("Snapshot angefordert.")
        self.show_status("Snapshot wird gespeichert...", level="busy")
        if self.last_frame is None:
            self.append_log("Snapshot abgebrochen: kein Kamerabild vorhanden.")
            self.show_status("Snapshot nicht moeglich: kein Kamerabild", level="warning", auto_reset_ms=3500)
            messagebox.showwarning("Kein Bild", "Es ist noch kein Kamerabild vorhanden.")
            return
        capture_dir = Path(__file__).resolve().parent / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        file_path = capture_dir / f"arducam_foto_{timestamp}.jpg"
        duplicate_counter = 1
        while file_path.exists():
            file_path = capture_dir / f"arducam_foto_{timestamp}_{duplicate_counter}.jpg"
            duplicate_counter += 1
        if cv2.imwrite(str(file_path), self.last_frame):
            self.append_log(f"Snapshot gespeichert: {file_path}")
            self.show_status("Snapshot gespeichert", level="success", auto_reset_ms=3000)
        else:
            self.append_log(f"Snapshot konnte nicht gespeichert werden: {file_path}")
            self.show_status("Snapshot konnte nicht gespeichert werden", level="error", auto_reset_ms=4500)
            messagebox.showerror("Snapshot-Fehler", f"Datei konnte nicht gespeichert werden:\n{file_path}")

    def reset_cycle(self):
        self.side_evaluations = {1: None, 2: None, 3: None, 4: None}
        self.current_side = 1
        self.refresh_side_status_cards()
        self.update_side_indicator()
        self.update_reference_view()
        self.update_overall_status()
        self.show_analysis_placeholder()
        self.update_defect_view()
        self.append_log("Pruefzyklus wurde zurueckgesetzt.")
        self.show_status("Pruefzyklus zurueckgesetzt", level="info", auto_reset_ms=2500)

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_running = False
        self.save_session_state(log_errors=True)
        self.root.destroy()


def resolve_default_config():
    candidate = Path(__file__).resolve().parent / "config.example.json"
    return str(candidate) if candidate.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Jetson Orin YOLOv8 Vollstaendigkeitspruefung")
    parser.add_argument("--config", default=resolve_default_config(), help="Pfad zu einer JSON-Konfiguration")
    args = parser.parse_args()

    config = AppConfig()
    config_path = args.config
    if config_path:
        try:
            config = load_app_config(config_path)
        except Exception:
            config = AppConfig()

    root = tk.Tk()
    app = JetsonQCApp(root, config=config, config_path=config_path)
    root.mainloop()


if __name__ == "__main__":
    main()
