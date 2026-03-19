import cv2
import time
import os

def live_stream_and_capture(save_directory="."):
    # 1. Kamera initialisieren (Video4Linux2 Backend für den Jetson)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden. Bitte USB-Verbindung prüfen.")
        return

    # 2. Auflösung einstellen (Passe diese Werte an deine Arducam an)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Kamera ist aktiv!")
    print("-----------------------------------------")
    print("Drücke 's' auf der Tastatur, um ein Foto zu speichern.")
    print("Drücke 'q' auf der Tastatur, um das Programm zu beenden.")
    print("-----------------------------------------")

    # 3. Endlos-Schleife für das Live-Bild
    while True:
        # Ein einzelnes Bild (Frame) von der Kamera lesen
        ret, frame = cap.read()

        if not ret:
            print("Fehler beim Lesen des Bildes von der Kamera.")
            break

        # 4. Das Bild in einem Fenster anzeigen
        cv2.imshow("Arducam Live-Vorschau", frame)

        # 5. Tastatureingabe abfragen (Wartet 1 Millisekunde pro Durchlauf)
        key = cv2.waitKey(1) & 0xFF

        # Wenn 's' gedrückt wird: Bild speichern
        if key == ord('s'):
            timestamp = int(time.time())
            filename = os.path.join(save_directory, f"arducam_foto_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            
            # Kurze Bestätigung im Terminal
            print(f"KLICK! Foto gespeichert unter: {filename}")

        # Wenn 'q' gedrückt wird: Schleife abbrechen und beenden
        elif key == ord('q'):
            print("Programm wird beendet...")
            break

    # 6. Am Ende aufräumen: Kamera freigeben und alle Fenster schließen
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_stream_and_capture()
