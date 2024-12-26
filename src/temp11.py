import cv2

# Calea către imagine și fișierul YOLO
image_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\output\yolo_t1_dataset\images\test\dad_0001.jpg"
annotations = [
    "0 0.698958 0.394444 0.389583 0.633333",
    "0 0.346875 0.472222 0.293750 0.333333",
    "0 0.271875 0.583333 0.139583 0.144444",
]

# Citirea imaginii
image = cv2.imread(image_path)
height, width, _ = image.shape

# Parsarea adnotărilor și desenarea dreptunghiurilor
for annotation in annotations:
    parts = annotation.split()
    class_id, center_x, center_y, w, h = map(float, parts)
    center_x, center_y, w, h = center_x * width, center_y * height, w * width, h * height

    # Calcularea colțurilor dreptunghiului
    x1 = int(center_x - w / 2)
    y1 = int(center_y - h / 2)
    x2 = int(center_x + w / 2)
    y2 = int(center_y + h / 2)

    # Desenarea dreptunghiului
    color = (0, 255, 0)  # Verde
    thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Afișarea imaginii cu dreptunghiurile YOLO
cv2.imshow("YOLO Annotations", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
