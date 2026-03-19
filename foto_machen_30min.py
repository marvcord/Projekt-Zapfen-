import cv2
import time
import os
import numpy as np

def foto_machen_30min(save_directory="."):
    # Kamera initialisieren
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden.")
        return

    # Auflösung einstellen
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Variablen für die Steuerung
    automatik_aktiv = False
    intervall_sekunden = 30 * 60
    letzte_aufnahme_zeit = 0
    start_zeit_24h = 0
    dauer_24h_sekunden = 24 * 3600
    
    # Neu: Foto-Zähler
    foto_anzahl = 0

    # Variablen für den Dunkelmodus
    dunkel_modus = False
    inaktiv_limit = 20 * 60  # 20 Minuten
    letzte_aktivitaet = time.time()
    
    # Schwarzes Bild für den Dunkelmodus erstellen
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Schrift-Einstellungen für die Texteinblendung
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_color = (255, 255, 255)  # Weiß
    thickness = 2
    position = (30, 50)  # Position oben links

    print("--- Programm 'foto_machen_30min' gestartet ---")
    print("Bedienung:")
    print("  's'          -> Automatisierung (24h) STARTEN")
    print("  'Enter'      -> Bildschirm aufwecken")
    print("  'q'          -> Programm BEENDEN")
    print("-------------------------------------------")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            jetzt = time.time()

            # Text für die Einblendung vorbereiten
            status_text = f"Fotos aufgenommen: {foto_anzahl}"
            if automatik_aktiv:
                status_text += " (Aktiv)"
            else:
                status_text += " (Bereit - Druecke 's')"

            # Inaktivitäts-Check
            if jetzt - letzte_aktivitaet > inaktiv_limit:
                dunkel_modus = True

            # Anzeige-Logik mit Texteinblendung
            if dunkel_modus:
                # Kopie des schwarzen Bildes, damit der Text nicht "einbrennt"
                display_frame = black_frame.copy()
                cv2.putText(display_frame, status_text, position, font, font_scale, font_color, thickness)
                cv2.putText(display_frame, "STANDBY - Enter zum Aufwecken", (30, 100), font, 0.7, (0, 255, 0), 1)
                cv2.imshow("Arducam - Standby", display_frame)
            else:
                # Text in das Live-Bild einblenden
                display_frame = frame.copy()
                cv2.putText(display_frame, status_text, position, font, font_scale, (0, 255, 0), thickness)
                cv2.imshow("Arducam - Live", display_frame)
            
            # Tastatur abfragen
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Programm wird beendet...")
                break

            elif key == ord('s'):
                letzte_aktivitaet = jetzt
                if not automatik_aktiv:
                    automatik_aktiv = True
                    start_zeit_24h = jetzt
                    letzte_aufnahme_zeit = jetzt - intervall_sekunden
                    print(f"[{time.ctime()}] >>> Automatisierung GESTARTET.")

            elif key == 13: # Enter Taste
                letzte_aktivitaet = jetzt
                if dunkel_modus:
                    dunkel_modus = False
                    print(f"[{time.ctime()}] Bildschirm aktiviert.")

            elif key != 255:
                letzte_aktivitaet = jetzt

            # --- Automatik-Logik (Fotos machen) ---
            if automatik_aktiv:
                if jetzt - start_zeit_24h > dauer_24h_sekunden:
                    print("--- 24 Stunden abgelaufen. Automatik beendet. ---")
                    automatik_aktiv = False
                    continue

                if jetzt - letzte_aufnahme_zeit >= intervall_sekunden:
                    # Wir speichern das "saubere" Original-Frame ohne Text-Einblendung!
                    timestamp = int(jetzt)
                    filename = os.path.join(save_directory, f"arducam_foto_{timestamp}.jpg")
                    
                    if cv2.imwrite(filename, frame):
                        foto_anzahl += 1 # Zähler erhöhen
                        letzte_aufnahme_zeit = jetzt
                        print(f"KLICK! Foto Nr. {foto_anzahl} gespeichert: {filename}")

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    foto_machen_30min()
