import cv2
import os
import random
import numpy as np

def extract_and_rotate_faces(image_path, bounding_boxes, output_dir, n_rotations=3):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imaginea nu a fost găsită la calea: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    def rotate_image(face, angle):
        (h, w) = face.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    for i, (_, x_min, y_min, x_max, y_max, label) in enumerate(bounding_boxes):
        face = image[y_min:y_max, x_min:x_max]

        face_path = os.path.join(output_dir, f"face_{i+1}_{label}.jpg")
        cv2.imwrite(face_path, face)

        for r in range(n_rotations):
            angle = random.randint(10, 30)
            rotated_face = rotate_image(face, angle)
            rotated_path = os.path.join(output_dir, f"face_{i+1}_{label}_rot{r+1}.jpg")
            cv2.imwrite(rotated_path, rotated_face)
            print(f"Fața {i+1}: rotire {r+1} ({angle}°) salvată la {rotated_path}")

    print("Procesarea a fost finalizată!")


# EXEMPLU DE FOLOSIRE
image_path = r'C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\dad\0001.jpg'
bounding_boxes = [
    ("0001.jpg", 242, 28, 429, 256, "dad"),
    ("0001.jpg", 96, 110, 237, 230, "unknown"),
    ("0001.jpg", 97, 184, 164, 236, "unknown")
]
output_dir = "extracted_faces"

extract_and_rotate_faces(image_path, bounding_boxes, output_dir, n_rotations=3)
