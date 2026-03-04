import cv2
import time
import os

def capture_and_save_image(save_directory="."):
    # 1. Kamera initialisieren
    # 0 ist meistens die erste angeschlossene USB-Kamera (/dev/video0).
    # cv2.CAP_V4L2 zwingt OpenCV, das Linux-spezifische Video4Linux2-Backend zu nutzen.
    # Das ist auf dem Jetson Orin wichtig für eine saubere USB-Kamera-Steuerung.
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden. Bitte USB-Verbindung prüfen.")
        return

    # 2. Kameraeinstellungen vornehmen
    # WICHTIG: Welche Parameter funktionieren, hängt exakt von deinem Arducam-Modell ab.
    
    # Auflösung einstellen (z.B. Full HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Bildrate (FPS) einstellen
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Helligkeit und Kontrast anpassen (Werte sind beispielhaft)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    # cap.set(cv2.CAP_PROP_CONTRAST, 128)

    # Automatische Belichtung ausschalten (1 = Manuell, 3 = Auto) und manuell setzen
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    # cap.set(cv2.CAP_PROP_EXPOSURE, 150) 

    # 3. Dem Sensor Zeit geben
    # Wenn die Kamera startet und Einstellungen geändert werden, braucht der 
    # Sensor kurz Zeit, um sich an die Lichtverhältnisse anzupassen.
    print("Kamera wird initialisiert... Bitte lächeln!")
    time.sleep(2.0)

    # Um den Puffer zu leeren, lesen wir ein paar Frames im Leerlauf (optional, aber empfohlen)
    for _ in range(5):
        cap.read()

    # 4. Das eigentliche Bild aufnehmen
    ret, frame = cap.read()

    if ret:
        # 5. Bild speichern
        # Generiert einen eindeutigen Dateinamen basierend auf der aktuellen Zeit
        timestamp = int(time.time())
        filename = os.path.join(save_directory, f"arducam_bild_{timestamp}.jpg")
        
        cv2.imwrite(filename, frame)
        print(f"Erfolg: Das Bild wurde erfolgreich als '{filename}' gespeichert.")
    else:
        print("Fehler: Konnte kein Bild vom Sensor lesen.")

    # 6. Ressourcen ordnungsgemäß freigeben
    cap.release()

if __name__ == "__main__":
    capture_and_save_image()
