import os
import cv2

# Calea către directorul principal (înlocuiește cu calea ta)
base_dir = "output/hard-negative/ratio_14"

# Dimensiunea la care dorim să redimensionăm imaginile
resize_to = (69, 49)

def resize_images_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Verifică dacă fișierul este o imagine
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(root, file)
                try:
                    # Încarcă imaginea
                    img = cv2.imread(image_path)
                    if img is not None:
                        # Redimensionează imaginea
                        resized_img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
                        # Salvează imaginea redimensionată peste cea originală
                        cv2.imwrite(image_path, resized_img)
                        print(f"Redimensionat: {image_path}")
                    else:
                        print(f"Imagine invalidă: {image_path}")
                except Exception as e:
                    print(f"Eroare la redimensionare: {image_path} - {e}")

# Rulează funcția pe directorul principal
resize_images_in_folder(base_dir)