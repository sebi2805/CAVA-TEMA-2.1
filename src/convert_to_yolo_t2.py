import os
import shutil

# Mapare clase -> ID YOLO
class_map = {
    "dad": 0,
    "mom": 1,
    "dexter": 2,
    "deedee": 3
}

# Conversie coordonate în format YOLO
def convert_to_yolo(image_width, image_height, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    width    = (xmax - xmin) / image_width
    height   = (ymax - ymin) / image_height
    return x_center, y_center, width, height

# Subfoldere și fișiere de adnotări
subfolders        = ["dad", "deedee", "dexter", "mom"]
annotations_files = ["dad_annotations.txt", 
                     "deedee_annotations.txt", 
                     "dexter_annotations.txt", 
                     "mom_annotations.txt"]

output_folder = "yolo_dataset_t2"

# Asigurăm directorul de ieșire curat
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

offset = 0  # pentru redenumirea imaginilor ca să nu se calce unele pe altele

for subfolder, annotations_file in zip(subfolders, annotations_files):
    image_folder = os.path.join(
        r'C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare',
        subfolder
    )
    ann_file_path = os.path.join(
        r'C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare',
        annotations_file
    )
    
    if not os.path.isfile(ann_file_path):
        print(f"Fișierul {ann_file_path} nu există! Trecem mai departe...")
        continue

    # Citim fișierul de adnotări și strângem toate bbox-urile per imagine
    images_info = {}  # ex: images_info["0999.jpg"] = [(xmin, ymin, xmax, ymax, class_name), ...]
    with open(ann_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                # Linia are un format ciudat sau incomplet?
                # (img, xmin, ymin, xmax, ymax, class_name)
                print(f"Linie nevalidă: {line}")
                continue

            image_name = parts[0]
            xmin       = int(parts[1])
            ymin       = int(parts[2])
            xmax       = int(parts[3])
            ymax       = int(parts[4])
            class_name = parts[5]

            if class_name not in class_map:
                print(f"Clasa {class_name} nu e definită în class_map. Omit.")
                continue

            if image_name not in images_info:
                images_info[image_name] = []
            images_info[image_name].append((xmin, ymin, xmax, ymax, class_name))

    # Acum parcurgem fiecare imagine și creăm fișierele YOLO
    sorted_images = sorted(images_info.keys())  # sortăm pentru consistență
    for img_idx, img_name in enumerate(sorted_images):
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Imaginea {img_path} nu există. Se omite.")
            continue

        # În cazul tău, 480x360 e fix. Altfel, citește dimensiunile real-time.
        image_width, image_height = 480, 360

        # Determinăm noul nume de imagine
        new_image_name = f"{offset + img_idx:04d}.jpg"
        new_image_path = os.path.join(output_folder, new_image_name)
        shutil.copy(img_path, new_image_path)

        # Creăm fișierul YOLO .txt
        yolo_file_name = f"{os.path.splitext(new_image_name)[0]}.txt"
        yolo_file_path = os.path.join(output_folder, yolo_file_name)

        yolo_annotations = []
        for (xmin, ymin, xmax, ymax, class_name) in images_info[img_name]:
            class_id = class_map[class_name]
            x_c, y_c, w, h = convert_to_yolo(image_width, image_height,
                                             xmin, ymin, xmax, ymax)
            yolo_annotations.append(
                f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            )

        with open(yolo_file_path, "w") as yolo_file:
            yolo_file.write("\n".join(yolo_annotations))
    
    # Creștem offset-ul cu un pas suficient de mare (1000 în original)
    offset += 1000

print("Conversia s-a încheiat cu succes!")
