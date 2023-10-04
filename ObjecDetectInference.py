# Importamos librerias
from ultralytics import YOLO
import cv2
import math

# Modelo
modelGlass = YOLO("Modelos/Gafas.pt")
modelCap = YOLO("Modelos/Gorras.pt")

# Cap
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Clases
clsNameCap = ['Gafas', 'Sombrero', 'Abrigo', 'Camisa', 'Pantalones', 'Shorts', 'Falda', 'Vestido', 'Maleta', 'Zapato']
clsNameGlass = ['Gafas']

# Confidence
confidenceCap = 0.5
confidenceGlass = 0.5

# Umbral
confThresholdCap = 0.5
confThresholdGlass = 0.5

# Inference
while True:
    # Frames
    ret, frame = cap.read()

    # Cap & Glass Detect
    resultsCap = modelCap(frame, stream = True, imgsz = 640)
    resultsGlass = modelGlass(frame, stream = True, imgsz = 640)

    # Cap
    for resCap in resultsCap:
        # Boxes Cap
        boxesCap = resCap.boxes
        for boxCap in boxesCap:
            # Bounding box
            xi1, yi1, xf1, yf1 = boxCap.xyxy[0]
            xi1, yi1, xf1, yf1 = int(xi1), int(yi1), int(xf1), int(yf1)

            # Error < 0
            if xi1 < 0: xi1 = 0
            if yi1 < 0: yi1 = 0
            if xf1 < 0: xf1 = 0
            if yf1 < 0: yf1 = 0

            # Class
            clsCap = int(boxCap.cls[0])

            # Confidence
            confCap = math.ceil(boxCap.conf[0])

            if clsCap == 1:
                # Draw Cap
                cv2.rectangle(frame, (xi1, yi1), (xf1, yf1), (255, 255, 0), 2)
                cv2.putText(frame, f"{clsNameCap[clsCap]} {int(confCap * 100)}%", (xi1, yi1 - 20),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    # Glass
    for resGlass in resultsGlass:
        # Boxes Glass
        boxesGlass = resGlass.boxes
        for boxGlass in boxesGlass:
            # Bounding box
            xi2, yi2, xf2, yf2 = boxGlass.xyxy[0]
            xi2, yi2, xf2, yf2 = int(xi2), int(yi2), int(xf2), int(yf2)

            # Error < 0
            if xi2 < 0: xi2 = 0
            if yi2 < 0: yi2 = 0
            if xf2 < 0: xf2 = 0
            if yf2 < 0: yf2 = 0

            # Class
            clsGlass = int(boxGlass.cls[0])

            # Confidence
            confGlass = math.ceil(boxGlass.conf[0])

            # Draw Cap
            cv2.rectangle(frame, (xi2, yi2), (xf2, yf2), (255, 0, 255), 2)
            cv2.putText(frame, f"{clsNameGlass[clsGlass]} {int(confGlass * 100)}%", (xi2, yi2 - 20), cv2.FONT_HERSHEY_COMPLEX,1, (255, 0, 255), 2)

    # Show
    cv2.imshow("Accesories Detect", frame)

    # Close
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()