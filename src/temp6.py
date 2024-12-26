import os
import cv2 as cv
from collections import Counter

def print_shape_counts(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folderul specificat nu există: {folder_path}")
        return

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print(f"Nu există imagini în folderul: {folder_path}")
        return

    print(f"Verificăm {len(image_files)} imagini din folderul: {folder_path}")
    shape_counter = Counter()

    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        img = cv.imread(image_path)

        if img is None:
            print(f"Imaginea {image_name} nu a putut fi încărcată (poate fi coruptă).")
            continue

        shape_counter[img.shape] += 1

    print("\nDimensiuni și număr de imagini pentru fiecare dimensiune:")
    for shape, count in shape_counter.items():
        print(f"Shape: {shape}, Count: {count}")

# Calea către folderul tău
folder = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\output\hard-negative\ratio_08"
print_shape_counts(folder)
