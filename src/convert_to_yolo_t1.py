import os
import shutil

# Hărțire simplă fișier adnotări -> subfolder imagini
annotation_map = {
    "dad_annotations.txt": "dad",
    "deedee_annotations.txt": "deedee",
    "dexter_annotations.txt": "dexter",
    "mom_annotations.txt": "mom"
}

# Dimensiunile imaginilor (pentru normalizare YOLO)
IMG_WIDTH = 480
IMG_HEIGHT = 360

# Unde vom pune imaginile și fișierele YOLO
OUTPUT_DIR = "./output/yolo_t1_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clasa e unică: face => class_id = 0
CLASS_ID = 0  

# Ca să nu amestecăm contorul între foldere, îl ținem global
global_counter = 0

# Dictionar care reține dacă am redenumit deja o anumită imagine,
# pentru că pot exista mai multe box-uri pentru aceeași imagine
renamed_images = {}  # cheie: (subfolder, old_img_name) -> new_img_name

for annotation_file, subfolder in annotation_map.items():
    # Calea către fișierul de adnotări
    ann_path = os.path.join("antrenare", annotation_file)
    
    # Citim fiecare linie din fișierul curent
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            old_img_name = parts[0]  # ex: "0001.jpg"
            x_min, y_min, x_max, y_max = map(int, parts[1:5])
            # Ignorăm numele clasei existente din fișier și îl suprascriem cu "face"

            # Verificăm dacă am redenumit deja imaginea
            if (subfolder, old_img_name) not in renamed_images:
                global_counter += 1
                new_img_name = f"{subfolder}_{global_counter:04d}.jpg"
                renamed_images[(subfolder, old_img_name)] = new_img_name

                # Copiem imaginea în OUTPUT_DIR
                src_img_path = os.path.join("antrenare", subfolder, old_img_name)
                dst_img_path = os.path.join(OUTPUT_DIR, new_img_name)
                shutil.copy2(src_img_path, dst_img_path)

            # Aflăm noul nume pentru fișier
            new_img_name = renamed_images[(subfolder, old_img_name)]
            txt_name = new_img_name.replace(".jpg", ".txt")
            txt_path = os.path.join(OUTPUT_DIR, txt_name)

            # Calculăm coordonatele normalizate
            x_center = ((x_min + x_max) / 2) / IMG_WIDTH
            y_center = ((y_min + y_max) / 2) / IMG_HEIGHT
            w = (x_max - x_min) / IMG_WIDTH
            h = (y_max - y_min) / IMG_HEIGHT

            # Scriem (append) coordonatele YOLO în fișierul text
            with open(txt_path, "a") as txt_file:
                txt_file.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("Conversie finalizată! Imaginile și fișierele YOLO sunt în folderul:", OUTPUT_DIR)
