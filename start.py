from ultralytics import YOLO

# 1. Modell laden
model = YOLO("yolov8n.pt")

# 2. Festlegen, was wir suchen (Beispiel: 'cell phone' oder 'cup')
# Tipp: YOLOv8 erkennt standardmäßig 80 Klassen (person, car, bottle, etc.)
GESUCHTER_GEGENSTAND = "cell phone"

# 3. Webcam Analyse (wir schauen uns nur 1 Bild an oder nutzen einen Loop)
# stream=True ist effizienter für Live-Vergleiche
results = model.predict(source="0", show=True, stream=True)

print(f"Suche nach: {GESUCHTER_GEGENSTAND}...")

for r in results:
    found_classes = []
    
    # Alle erkannten Objekte in diesem Frame durchgehen
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        found_classes.append(label)

    # VERGLEICH: Ist unser Wunsch-Gegenstand dabei?
    if GESUCHTER_GEGENSTAND in found_classes:
        print(f">>> OK: {GESUCHTER_GEGENSTAND} erkannt! Übereinstimmung gefunden.")
    else:
        print("... Suche läuft (Gegenstand nicht im Bild) ...")