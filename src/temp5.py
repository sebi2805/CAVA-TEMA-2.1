import cv2
import numpy as np

# Funcție pentru a calcula IoU între două bounding box-uri
def calculate_iou(box1, box2):
    # Box-urile sunt sub formă de (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculăm aria de intersecție
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculăm aria fiecărui bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculăm aria de uniune
    union = box1_area + box2_area - intersection

    # IoU = Intersecție / Uniune
    iou = intersection / union if union > 0 else 0
    return iou

# Callback pentru a desena bounding box-uri
boxes = []
def draw_box(event, x, y, flags, param):
    global boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        boxes.append((x, y))  # Începutul box-ului
    elif event == cv2.EVENT_LBUTTONUP:
        boxes[-1] = boxes[-1] + (x, y)  # Sfârșitul box-ului

# Imagine blank pentru desen
image = cv2.imread(r'C:\Users\User\Desktop\university\CAVA-TEMA-2\output\metrics\detections\076.jpg')

cv2.namedWindow("Select Bounding Boxes")
cv2.setMouseCallback("Select Bounding Boxes", draw_box)

print("Instrucțiuni:")
print("1. Desenează două bounding box-uri apăsând și trăgând cu mouse-ul.")
print("2. După desen, apasă 'c' pentru a calcula IoU sau 'q' pentru a ieși.")

while True:
    temp_image = image.copy()

    # Desenăm toate box-urile curente
    for box in boxes:
        if len(box) == 4:  # Dacă box-ul este complet (are 4 coordonate)
            cv2.rectangle(temp_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Select Bounding Boxes", temp_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and len(boxes) >= 2:
        if len(boxes) > 2:
            print("Au fost selectate mai mult de 2 box-uri, se vor folosi primele 2.")
        box1 = boxes[0]
        box2 = boxes[1]
        iou = calculate_iou(box1, box2)
        print(f"Bounding Box 1: {box1}")
        print(f"Bounding Box 2: {box2}")
        print(f"IoU: {iou:.4f}")
    elif key == ord('q'):  # Apasă 'q' pentru a ieși
        break

cv2.destroyAllWindows()
