import cv2
import math
import time
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

# --- KONFIGURATION ---
MODEL_PATH = 'best.pt'  # Pfad zu deinem trainierten Jetson-Modell
CAMERA_ID = 0           # 0 für USB-Cam

CLASS_NAMES = {
    0: "Feder",
    1: "Führungsrolle",
    2: "Sicherungsring_Fr",
    3: "Laufrolle_schmal",
    4: "Sicherungsring_Lrs",
    5: "Laufrolle_vorne",
    6: "Sicherungsring_Lrv",
    7: "Laufrolle_hinten",
    8: "Sicherungsring_Lrh",
    9: "Gussteil_gross",
    10: "Gussteil_klein"
}

# Soll-Zustand (Golden Sample) aus deinen Daten
EXPECTED_STATE = {
    1: [
        {'class': 9, 'x': 0.425481, 'y': 0.599265, 'tol': 0.05},
        {'class': 9, 'x': 0.332761, 'y': 0.244026, 'tol': 0.05},
        {'class': 0, 'x': 0.530563, 'y': 0.357077, 'tol': 0.05},
        {'class': 6, 'x': 0.669299, 'y': 0.715074, 'tol': 0.05},
        {'class': 6, 'x': 0.673764, 'y': 0.724724, 'tol': 0.05},
        {'class': 7, 'x': 0.713255, 'y': 0.303309, 'tol': 0.05},
        {'class': 8, 'x': 0.717720, 'y': 0.323529, 'tol': 0.05},
        {'class': 10, 'x': 0.668269, 'y': 0.514706, 'tol': 0.05},
        {'class': 10, 'x': 0.466690, 'y': 0.763787, 'tol': 0.05}
    ],
    2: [
        {'class': 10, 'x': 0.462569, 'y': 0.461397, 'tol': 0.05},
        {'class': 9, 'x': 0.262363, 'y': 0.460478, 'tol': 0.05},
        {'class': 1, 'x': 0.616415, 'y': 0.684283, 'tol': 0.05},
        {'class': 5, 'x': 0.618132, 'y': 0.259191, 'tol': 0.05}
    ],
    3: [
        {'class': 10, 'x': 0.464286, 'y': 0.463695, 'tol': 0.05},
        {'class': 9, 'x': 0.262706, 'y': 0.463235, 'tol': 0.05},
        {'class': 1, 'x': 0.617102, 'y': 0.682445, 'tol': 0.05},
        {'class': 5, 'x': 0.620536, 'y': 0.261949, 'tol': 0.05}
    ],
    4: [
        {'class': 10, 'x': 0.461538, 'y': 0.463235, 'tol': 0.05},
        {'class': 9, 'x': 0.261676, 'y': 0.464614, 'tol': 0.05},
        {'class': 1, 'x': 0.618475, 'y': 0.682904, 'tol': 0.05},
        {'class': 5, 'x': 0.626030, 'y': 0.254136, 'tol': 0.05}
    ]
}

class QualityControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nvidia Jetson Orin - Bauteilkontrolle")
        self.root.geometry("1200x750")
        self.root.configure(bg="#2c3e50")

        # Erstelle Ordner für Trainingsdaten, falls nicht vorhanden
        os.makedirs("dataset/images/train", exist_ok=True)
        os.makedirs("dataset/labels/train", exist_ok=True)

        # Modell laden
        try:
            self.model = YOLO(MODEL_PATH)
            self.log_msg = "System bereit. Modell geladen."
        except:
            self.model = None
            self.log_msg = "WARNUNG: Modell nicht gefunden! Nur UI-Modus."

        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.current_side = 1
        self.results_data = {1: [], 2: [], 3: [], 4: []}

        self.setup_ui()
        self.update_video()

    def setup_ui(self):
        # Haupt-Container
        self.main_frame = tk.Frame(self.root, bg="#2c3e50")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Video-Anzeige
        self.video_label = tk.Label(self.main_frame, bg="black", borderwidth=2, relief="solid")
        self.video_label.pack(side=tk.LEFT, padx=10)

        # Kontroll-Panel
        self.control_panel = tk.Frame(self.main_frame, bg="#34495e", width=400)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.status_title = tk.Label(self.control_panel, text="PRÜFUNG", font=("Helvetica", 20, "bold"), bg="#34495e", fg="white")
        self.status_title.pack(pady=10)

        self.side_label = tk.Label(self.control_panel, text=f"AKTUELL: SEITE {self.current_side}", font=("Helvetica", 14), bg="#34495e", fg="#ecf0f1")
        self.side_label.pack(pady=5)

        # Prüfen Button
        self.btn_check = tk.Button(self.control_panel, text="JETZT PRÜFEN (Leertaste)", font=("Helvetica", 12, "bold"), 
                                  bg="#27ae60", fg="white", command=self.capture_side, height=2)
        self.btn_check.pack(pady=10, fill=tk.X, padx=20)

        # Neustart Button
        self.btn_reset = tk.Button(self.control_panel, text="Neustart / Nächstes Teil", command=self.reset_app, bg="#c0392b", fg="white", font=("Helvetica", 10, "bold"))
        self.btn_reset.pack(pady=5, fill=tk.X, padx=20)

        # --- NEU: Data Collection Button ---
        self.btn_train_data = tk.Button(self.control_panel, text="📸 Trainingsbild speichern", command=self.save_training_data, bg="#2980b9", fg="white", font=("Helvetica", 10, "bold"))
        self.btn_train_data.pack(pady=15, fill=tk.X, padx=20)

        self.log_text = tk.Text(self.control_panel, height=20, width=40, bg="#212f3c", fg="#ecf0f1", font=("Consolas", 10))
        self.log_text.pack(pady=10, padx=10)
        self.log(self.log_msg)

        # Hotkeys binden
        self.root.bind('<space>', lambda e: self.capture_side())
        self.root.bind('s', lambda e: self.save_training_data()) # Mit 's' schnell ein Trainingsbild speichern

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (720, 540))
            cv2image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(15, self.update_video)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def save_training_data(self):
        """Speichert das aktuelle Bild und erstellt eine YOLO Auto-Label Datei."""
        ret, frame = self.cap.read()
        if not ret:
            self.log("Fehler beim Zugriff auf die Kamera!")
            return

        # Generiere Dateinamen basierend auf der aktuellen Zeit
        timestamp = int(time.time())
        base_filename = f"arducam_foto_{timestamp}"
        img_path = f"dataset/images/train/{base_filename}.jpg"
        txt_path = f"dataset/labels/train/{base_filename}.txt"

        # 1. Bild speichern
        cv2.imwrite(img_path, frame)
        
        # 2. Label-Datei erstellen (Auto-Labeling)
        label_lines = []
        if self.model is not None:
            results = self.model(frame, verbose=False)
            img_h, img_w, _ = frame.shape
            
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    # Berechne YOLO Standard-Format: x_center, y_center, width, height
                    x_c = ((box.xyxy[0][0] + box.xyxy[0][2]) / 2) / img_w
                    y_c = ((box.xyxy[0][1] + box.xyxy[0][3]) / 2) / img_h
                    w = (box.xyxy[0][2] - box.xyxy[0][0]) / img_w
                    h = (box.xyxy[0][3] - box.xyxy[0][1]) / img_h
                    
                    label_lines.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # Textdatei schreiben (falls das Modell nichts findet, wird eine leere Datei erstellt, was auch okay ist)
        with open(txt_path, "w") as f:
            f.write("\n".join(label_lines))

        self.log(f"\n[DATEN] Gespeichert: {base_filename}")
        if self.model:
            self.log(f"-> Auto-Labeling: {len(label_lines)} Bauteile erkannt.")
        else:
            self.log("-> Kein Modell geladen, leere txt-Datei erstellt.")

    def capture_side(self):
        if self.current_side > 4: return

        ret, frame = self.cap.read()
        if not ret or self.model is None: return

        # YOLO Analyse
        results = self.model(frame, verbose=False)
        found_in_step = []
        img_h, img_w, _ = frame.shape

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x_c = ((box.xyxy[0][0] + box.xyxy[0][2]) / 2) / img_w
                y_c = ((box.xyxy[0][1] + box.xyxy[0][3]) / 2) / img_h
                found_in_step.append({'class': cls, 'x': float(x_c), 'y': float(y_c)})

        self.results_data[self.current_side] = found_in_step
        self.log(f"Seite {self.current_side}: {len(found_in_step)} Bauteile erkannt.")
        
        self.current_side += 1
        if self.current_side <= 4:
            self.side_label.config(text=f"AKTUELL: SEITE {self.current_side}")
        else:
            self.side_label.config(text="PRÜFUNG ABGESCHLOSSEN", fg="#f1c40f")
            self.btn_check.config(state=tk.DISABLED)
            self.final_eval()

    def final_eval(self):
        self.log("\n--- ANALYSE ERGEBNIS ---")
        overall_ok = True
        
        for side, expected_list in EXPECTED_STATE.items():
            detected_list = self.results_data[side]
            
            for target in expected_list:
                t_cls = target['class']
                t_name = CLASS_NAMES.get(t_cls, "Unbekannt")
                
                candidates = [d for d in detected_list if d['class'] == t_cls]
                
                if not candidates:
                    self.log(f"FEHLER: {t_name} auf Seite {side} fehlt!")
                    overall_ok = False
                    continue
                
                at_correct_pos = False
                for cand in candidates:
                    dist = math.sqrt((cand['x'] - target['x'])**2 + (cand['y'] - target['y'])**2)
                    if dist <= target['tol']:
                        at_correct_pos = True
                        break
                
                if not at_correct_pos:
                    self.log(f"FEHLER: {t_name} auf Seite {side} falsch platziert!")
                    overall_ok = False

        if overall_ok:
            self.log("\n>>> STATUS: TEIL VOLLSTÄNDIG & KORREKT")
            messagebox.showinfo("Ergebnis", "Bauteil ist in Ordnung (I.O.)")
        else:
            self.log("\n>>> STATUS: GEPRÜFT MIT FEHLERN")
            messagebox.showerror("Ergebnis", "Bauteil fehlerhaft (N.I.O.)")

    def reset_app(self):
        self.current_side = 1
        self.results_data = {1: [], 2: [], 3: [], 4: []}
        self.side_label.config(text=f"AKTUELL: SEITE {self.current_side}", fg="#ecf0f1")
        self.btn_check.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log("System zurückgesetzt.")

if __name__ == "__main__":
    root = tk.Tk()
    app = QualityControlApp(root)
    root.mainloop()
