import cv2

# Calea către imagine și fișierul YOLO
image_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\output\yolo_t2_dataset\images\test\0002.jpg"
annotations = [
    "2 0.419792 0.673611 0.168750 0.169444",
    "0 0.783333 0.226389 0.112500 0.241667"
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
