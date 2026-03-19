
import cv2
import os
import glob
import shutil

# Konfiguration (Pfade)
BILD_ORDNER = "C:/Users/marvm/Documents/Techniker/M7 KOF/Bilder_roh"
ZIEL_ORDNER = "C:/Users/marvm/Documents/Techniker/M7 KOF/Bilder_fertig"

# --- KLASSEN FÜR YOLO ---
KLASSEN = {
    ord('0'): (0, "Feder"),
    ord('1'): (1, "Fuehrungsrolle"),
    ord('2'): (2, "Sicherungsring_Fr"),
    ord('3'): (3, "Laufrolle_schmal"),
    ord('4'): (4, "Sicherungsring_Lrs"),
    ord('5'): (5, "Laufrolle_vorne"),
    ord('6'): (6, "Sicherungsring_Lrv"),
    ord('7'): (7, "Laufrolle_hinten"),
    ord('8'): (8, "Sicherungsring_Lrh"),
    ord('9'): (9, "Gussteil_gross"),
    ord('k'): (10, "Gussteil_klein"),
}

if not os.path.exists(ZIEL_ORDNER):
    os.makedirs(ZIEL_ORDNER)

# Globale Variablen
drawing = False
ix, iy = -1, -1
img_original = None  # NEU: Das saubere Originalbild
img_display = None   # Das Bild, das angezeigt wird
annotationen = []
warten_auf_taste = False
temp_box = ()

def redraw_image():
    """Baut das Anzeigebild aus dem Original und den verbleibenden Markierungen neu auf."""
    global img_display
    img_display = img_original.copy()
    
    for ann in annotationen:
        teile = ann.split()
        # ID, x_c, y_c, w, h, x_min, y_min, x_max, y_max
        kl_id = int(teile[0])
        x_min, y_min, x_max, y_max = int(teile[5]), int(teile[6]), int(teile[7]), int(teile[8])
        
        # Den Namen zur ID wiederfinden
        kl_name = "Unbekannt"
        for k, v in KLASSEN.items():
            if v[0] == kl_id:
                kl_name = v[1]
                break
                
        # Rahmen und Text neu zeichnen
        cv2.rectangle(img_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img_display, kl_name, (x_min, y_min - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img_display, annotationen, warten_auf_taste, temp_box

    if event == cv2.EVENT_LBUTTONDOWN:
        if not warten_auf_taste: 
            drawing = True
            ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_display.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('KI Label Tool', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            
            x_min, x_max = min(ix, x), max(ix, x)
            y_min, y_max = min(iy, y), max(iy, y)
            temp_box = (x_min, y_min, x_max, y_max)
            
            warten_auf_taste = True

def main():
    global img_original, img_display, annotationen, warten_auf_taste, temp_box
    
    bild_pfade = glob.glob(os.path.join(BILD_ORDNER, "*.jpg"))
    if not bild_pfade:
        print("Keine Bilder gefunden!")
        return

    cv2.namedWindow('KI Label Tool')
    cv2.setMouseCallback('KI Label Tool', mouse_callback)

    for pfad in bild_pfade:
        # Originalbild laden und saubere Kopie für die Anzeige machen
        img_original = cv2.imread(pfad)
        img_display = img_original.copy()
        img_h, img_w = img_display.shape[:2]
        
        annotationen = []
        warten_auf_taste = False
        
        print(f"\n--- Öffne: {os.path.basename(pfad)} ---")
        
        while True:
            temp_view = img_display.copy()
            if warten_auf_taste:
                cv2.putText(temp_view, "Warte auf Taste (0-9/k)... oder 'z' für Abbruch", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('KI Label Tool', temp_view)
            key = cv2.waitKey(50) & 0xFF
            
            # --- MODUS 1: Zuweisen oder Abbrechen ---
            if warten_auf_taste:
                if key in KLASSEN:
                    klasse_id, klasse_name = KLASSEN[key]
                    x_min, y_min, x_max, y_max = temp_box
                    
                    cv2.putText(img_display, klasse_name, (x_min, y_min - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # YOLO-Berechnung
                    x_center = ((x_min + x_max) / 2.0) / img_w
                    y_center = ((y_min + y_max) / 2.0) / img_h
                    box_w = (x_max - x_min) / img_w
                    box_h = (y_max - y_min) / img_h
                    
                    line = f"{klasse_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f} {x_min} {y_min} {x_max} {y_max}"
                    annotationen.append(line)
                    print(f"Gespeichert: {klasse_name}")
                    
                    warten_auf_taste = False 
                    
                # NEU: Abbrechen, wenn noch keine Taste zugewiesen wurde
                elif key == ord('z') or key == 27: # 27 ist die ESC-Taste
                    warten_auf_taste = False
                    redraw_image() # Entfernt die ungespeicherte Box aus der Ansicht
                    print("Zeichnen abgebrochen.")
            
            # --- MODUS 2: Normaler Modus (Nächstes Bild, Zurück, Beenden) ---
            else:
                # NEU: Letzte gespeicherte Markierung löschen
                if key == ord('z'):
                    if len(annotationen) > 0:
                        entfernt = annotationen.pop()
                        redraw_image()
                        print("Letzte Markierung rückgängig gemacht!")
                    else:
                        print("Keine Markierungen vorhanden, die gelöscht werden könnten.")

                elif key == ord('n'):
                    dateiname_base = os.path.basename(pfad)
                    txt_name = dateiname_base.replace('.jpg', '.txt')
                    txt_pfad_komplett = os.path.join(ZIEL_ORDNER, txt_name)
                    ziel_bild_pfad = os.path.join(ZIEL_ORDNER, dateiname_base)
                    
                    try:
                        if annotationen:
                            with open(txt_pfad_komplett, 'w') as f:
                                for ann in annotationen:
                                    f.write(ann + "\n")
                            print(f"Daten gesichert: {txt_name}")
                        
                        if os.path.exists(pfad):
                            shutil.move(pfad, ziel_bild_pfad)
                            print(f"Bild verschoben nach: {ZIEL_ORDNER}")
                            
                    except Exception as e:
                        print(f"Fehler beim Speichern oder Verschieben: {e}")
                    
                    break
                
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()
    print("Alle Bilder bearbeitet!")

if __name__ == "__main__":
    main()
