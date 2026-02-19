import cv2
import numpy as np

# Funktion zum Aufnehmen eines Bildes von der Kamera
def capture_image():
    cap = cv2.VideoCapture(0)  # 0 für die erste Kamera

    if not cap.isOpened():
        print("Kann die Kamera nicht öffnen.")
        return

    ret, frame = cap.read()

    if ret:
        return frame
    else:
        print("Kein Bild aufgenommen.")
        return None

# Funktion um Bilder zu unterscheiden
def compare_images(base_image, new_image):
    # Konvertiere die Bilder in Graustufen
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Berechne die Differenz zwischen den Bildern
    difference = cv2.absdiff(base_gray, new_gray)

    # Setze einen Schwellwert, um zu sehen, ob die Bilder unterschiedlich sind
    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Zähle die Unterschiede
    non_zero_count = np.count_nonzero(thresh)

    return non_zero_count > 1000  # Anpassung des Schwellenwerts nach Bedarf

if __name__ == "__main__":
    # Baseline-Bild aufnehmen
    print("Basisbild aufnehmen...")
    base_image = capture_image()

    if base_image is not None:
        while True:
            print("Neues Bild aufnehmen...")
            new_image = capture_image()

            if new_image is not None:
                # Überprüfe, ob die Bilder unterschiedlich sind
                if compare_images(base_image, new_image):
                    print("Die Bilder sind unterschiedlich!")
                else:
                    print("Die Bilder sind gleich.")
                
                # Zeige das neue Bild an
                cv2.imshow('New Image', new_image)
                
                # Breche die Schleife, wenn die 'q' Taste gedrückt wird
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        
        



            
            
