import cv2
import math
import time
import os
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# --- KONFIGURATION ---
MODEL_PATH = 'best.pt' 
BASE_MODEL = 'yolo11n.pt' # Basismodell, falls 'best.pt' noch nicht existiert
CAMERA_ID = 0 
EPOCHS = 100 # Anzahl der Trainingsdurchläufe

CLASS_NAMES = {
    0: "Feder", 1: "Führungsrolle", 2: "Sicherungsring_Fr",
    3: "Laufrolle_schmal", 4: "Sicherungsring_Lrs", 5: "Laufrolle_vorne",
    6: "Sicherungsring_Lrv", 7: "Laufrolle_hinten", 8: "Sicherungsring_Lrh",
    9: "Gussteil_gross", 10: "Gussteil_klein"
}

class QualityControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jetson Orin QC - Inspektion & Training")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        # Speicher für Referenz und Ergebnisse
        self.reference_state = {1: [], 2: [], 3: [], 4: []}
        self.results_data = {1: [], 2: [], 3: [], 4: []}
        self.current_side = 1
        self.is_training = False # Verhindert, dass mehrmals trainiert wird

        # Ordnerstruktur für unendlich viele Trainingsdaten sicherstellen
        os.makedirs("dataset/images/train", exist_ok=True)
        os.makedirs("dataset/labels/train", exist_ok=True)

        self.load_ai_model()

        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.setup_ui()
        self.update_video()

    def load_ai_model(self):
        """Lädt das Modell in den Speicher."""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = YOLO(MODEL_PATH)
                self.log_msg = f"Erfolgreich geladen: {MODEL_PATH}"
            else:
                self.model = YOLO(BASE_MODEL)
                self.log_msg = f"Kein {MODEL_PATH} gefunden. Starte mit Basismodell {BASE_MODEL}."
        except Exception as e:
            self.model = None
            self.log_msg = f"WARNUNG: Modellfehler! {e}"

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#2c3e50")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.video_label = tk.Label(self.main_frame, bg="black")
        self.video_label.pack(side=tk.LEFT, padx=10)

        self.control_panel = tk.Frame(self.main_frame, bg="#34495e")
        self.control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # UI Elemente für die Prüfung
        self.btn_load_ref = tk.Button(self.control_panel, text="📁 Referenz-Dateien laden", 
                                     command=self.load_reference_files, bg="#f39c12", font=("Helvetica", 10, "bold"))
        self.btn_load_ref.pack(pady=10, fill=tk.X, padx=20)

        self.side_label = tk.Label(self.control_panel, text="Keine Referenz geladen", font=("Helvetica", 12), bg="#34495e", fg="#e74c3c")
        self.side_label.pack(pady=5)

        self.btn_check = tk.Button(self.control_panel, text="PRÜFEN (Leertaste)", state=tk.DISABLED,
                                  bg="#27ae60", fg="white", command=self.capture_side, font=("Helvetica", 12, "bold"))
        self.btn_check.pack(pady=10, fill=tk.X, padx=20)

        # Datensammlung
        self.btn_train_data = tk.Button(self.control_panel, text="📸 Neues Trainingsbild (s)", command=self.save_training_data, bg="#2980b9", fg="white")
        self.btn_train_data.pack(pady=5, fill=tk.X, padx=20)

        # --- NEU: Training Button ---
        self.btn_train_model = tk.Button(self.control_panel, text="🚀 Modell jetzt trainieren", command=self.start_training_thread, bg="#8e44ad", fg="white", font=("Helvetica", 10, "bold"))
        self.btn_train_model.pack(pady=20, fill=tk.X, padx=20)

        self.log_text = tk.Text(self.control_panel, height=18, width=40, bg="#212f3c", fg="#ecf0f1")
        self.log_text.pack(pady=10, padx=10)
        
        if hasattr(self, 'log_msg'):
            self.log(self.log_msg)

        self.root.bind('<space>', lambda e: self.capture_side())
        self.root.bind('s', lambda e: self.save_training_data())

    def load_reference_files(self):
        for side in range(1, 5):
            file_path = filedialog.askopenfilename(title=f"Wähle Referenz-Textdatei für SEITE {side}")
            if file_path:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                side_data = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        side_data.append({
                            'class': int(parts[0]),
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'tol': 0.05
                        })
                self.reference_state[side] = side_data
                self.log(f"Referenz Seite {side} geladen: {len(side_data)} Teile.")
        
        self.btn_check.config(state=tk.NORMAL)
        self.side_label.config(text=f"AKTUELL: SEITE {self.current_side}", fg="#ecf0f1")

    def capture_side(self):
        if self.is_training: return
        if not self.reference_state[1]:
            messagebox.showwarning("Fehler", "Bitte zuerst Referenz-Dateien laden!")
            return

        ret, frame = self.cap.read()
        if not ret or not self.model: return

        results = self.model(frame, verbose=False)
        found = []
        img_h, img_w, _ = frame.shape

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x_c = ((box.xyxy[0][0] + box.xyxy[0][2]) / 2) / img_w
                y_c = ((box.xyxy[0][1] + box.xyxy[0][3]) / 2) / img_h
                found.append({'class': cls, 'x': float(x_c), 'y': float(y_c)})

        self.results_data[self.current_side] = found
        self.log(f"Seite {self.current_side} erfasst.")
        
        self.current_side += 1
        if self.current_side > 4:
            self.final_eval()
        else:
            self.side_label.config(text=f"AKTUELL: SEITE {self.current_side}")

    def final_eval(self):
        self.log("\n--- AUSWERTUNG ---")
        errors = []
        for side, expected in self.reference_state.items():
            detected = self.results_data[side]
            for target in expected:
                name = CLASS_NAMES.get(target['class'], "Unbekannt")
                match = [d for d in detected if d['class'] == target['class']]
                
                if not match:
                    errors.append(f"Seite {side}: {name} FEHLT")
                    continue
                
                if not any(math.sqrt((m['x']-target['x'])**2 + (m['y']-target['y'])**2) <= target['tol'] for m in match):
                    errors.append(f"Seite {side}: {name} FALSCHE POSITION")

        if not errors:
            self.log("ERGEBNIS: I.O.")
            messagebox.showinfo("OK", "Bauteil vollständig!")
        else:
            for e in errors: self.log("-> " + e)
            messagebox.showerror("N.I.O.", "Fehler gefunden!")
        
        self.current_side = 1
        self.side_label.config(text=f"AKTUELL: SEITE {self.current_side}")

    def save_training_data(self):
        if self.is_training: return
        ret, frame = self.cap.read()
        if ret:
            ts = int(time.time())
            # Bild speichern
            img_path = f"dataset/images/train/img_{ts}.jpg"
            cv2.imwrite(img_path, frame)
            
            # Label speichern (falls Modell geladen)
            if self.model:
                results = self.model(frame, verbose=False)
                img_h, img_w, _ = frame.shape
                labels = []
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        x_c = ((box.xyxy[0][0] + box.xyxy[0][2]) / 2) / img_w
                        y_c = ((box.xyxy[0][1] + box.xyxy[0][3]) / 2) / img_h
                        w = (box.xyxy[0][2] - box.xyxy[0][0]) / img_w
                        h = (box.xyxy[0][3] - box.xyxy[0][1]) / img_h
                        labels.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                
                with open(f"dataset/labels/train/img_{ts}.txt", "w") as f:
                    f.write("\n".join(labels))
            
            self.log(f"Datensatz hinzugefügt: img_{ts}")

    # ==========================================
    # --- NEU: TRAININGSLOGIK ---
    # ==========================================
    def clean_labels_for_training(self):
        """Entfernt überflüssige Pixeldaten aus alten Textdateien."""
        label_dir = "dataset/labels/train"
        for filename in os.listdir(label_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(label_dir, filename)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                clean_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        clean_lines.append(" ".join(parts[:5])) # Nur die ersten 5 YOLO-Werte behalten
                with open(filepath, 'w') as f:
                    f.write("\n".join(clean_lines))

    def create_yaml_config(self):
        """Erstellt die Landkarte für das YOLO Training."""
        yaml_content = f"""
path: {os.path.abspath('dataset')}
train: images/train
val: images/train

names:
"""
        for class_id, class_name in CLASS_NAMES.items():
            yaml_content += f"  {class_id}: {class_name}\n"
            
        with open("bauteile_data.yaml", "w") as f:
            f.write(yaml_content)

    def start_training_thread(self):
        if self.is_training: return
        
        # Abfrage ob man wirklich starten will
        if not messagebox.askyesno("Training", "Möchtest du das Modell jetzt mit allen im Ordner liegenden Bildern trainieren? Das kann einige Zeit dauern."):
            return

        self.is_training = True
        self.log("\n--- TRAINING GESTARTET ---")
        self.btn_train_model.config(state=tk.DISABLED, text="Training läuft...")
        self.btn_check.config(state=tk.DISABLED)
        self.btn_train_data.config(state=tk.DISABLED)
        
        # Gebe das aktuelle Modell aus dem VRAM frei, damit Platz fürs Training ist
        self.model = None 
        
        # Starte den Prozess im Hintergrund
        threading.Thread(target=self.run_yolo_training, daemon=True).start()

    def run_yolo_training(self):
        try:
            self.log("1. Bereinige alte Textdateien...")
            self.clean_labels_for_training()
            
            self.log("2. Erstelle Trainings-Konfiguration...")
            self.create_yaml_config()
            
            self.log(f"3. Starte YOLO Training ({EPOCHS} Epochs). Bitte warten...")
            
            # Nutze best.pt zum Weiterlernen, falls es existiert, sonst yolo11n.pt
            model_to_train = MODEL_PATH if os.path.exists(MODEL_PATH) else BASE_MODEL
            training_model = YOLO(model_to_train)
            
            # GPU Training starten
            results = training_model.train(
                data="bauteile_data.yaml",
                epochs=EPOCHS,
                imgsz=640,
                device=0, # 0 = Jetson Orin GPU
                verbose=False
            )
            
            # Kopiere das beste neue Modell in das Hauptverzeichnis
            new_best_path = str(results.save_dir / "weights/best.pt")
            if os.path.exists(new_best_path):
                import shutil
                shutil.copy(new_best_path, MODEL_PATH)
            
            self.log("\n--- TRAINING ABGESCHLOSSEN ---")
            self.log("Neues Modell wurde erfolgreich integriert!")
            
        except Exception as e:
            self.log(f"\nFEHLER beim Training: {e}")
        
        finally:
            # Lade das Modell wieder in den Speicher für die Inspektion
            self.log("Lade KI-Modell wieder in den Prüf-Modus...")
            self.load_ai_model()
            
            self.is_training = False
            # UI Updates müssen im Haupt-Thread aufgerufen werden
            self.root.after(0, self.reset_ui_after_training)

    def reset_ui_after_training(self):
        self.btn_train_model.config(state=tk.NORMAL, text="🚀 Modell jetzt trainieren")
        self.btn_check.config(state=tk.NORMAL if self.reference_state[1] else tk.DISABLED)
        self.btn_train_data.config(state=tk.NORMAL)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (720, 540)), cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(15, self.update_video)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = QualityControlApp(root)
    root.mainloop()
