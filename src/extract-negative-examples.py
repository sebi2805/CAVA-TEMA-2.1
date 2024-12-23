import os
import cv2 as cv
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_folder = 'antrenare'
output_folder = 'output'
negative_folder = os.path.join(output_folder, 'negative')
characters = ['dad', 'mom', 'dexter', 'deedee']

if not os.path.exists(negative_folder):
    os.makedirs(negative_folder)
    logging.info(f"Created negative folder: {negative_folder}")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea == 0 or boxBArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def does_intersect_with_any(pos_boxes, neg_box, max_iou_allowed=0.0):
    for pb in pos_boxes:
        if iou(pb, neg_box) > max_iou_allowed:  
            return True
    return False

def load_annotations():
    annotations_dict = {}
    for character in characters:
        annotation_file = os.path.join(input_folder, f'{character}_annotations.txt')
        characters_images_folder = os.path.join(input_folder, character)
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                image_name = line[0]
                x_min, y_min, x_max, y_max = map(int, line[1:5])
                image_path = os.path.join(characters_images_folder, image_name)
                if image_path not in annotations_dict:
                    annotations_dict[image_path] = []
                annotations_dict[image_path].append((x_min, y_min, x_max, y_max))
    return annotations_dict

def create_pyramid(img, scale_factor=0.8, min_size=(32, 32)):
    pyramid, scales = [], []
    current_img = img.copy()
    current_scale = 1.0
    while current_img.shape[0] >= min_size[1] and current_img.shape[1] >= min_size[0]:
        pyramid.append(current_img)
        scales.append(current_scale)
        new_w = int(current_img.shape[1] * scale_factor)
        new_h = int(current_img.shape[0] * scale_factor)
        current_scale *= scale_factor
        current_img = cv.resize(img, (new_w, new_h))
    return pyramid, scales

def scale_bboxes(bboxes, scale):
    scaled = []
    for (x_min, y_min, x_max, y_max) in bboxes:
        scaled.append((
            int(x_min * scale),
            int(y_min * scale),
            int(x_max * scale),
            int(y_max * scale)
        ))
    return scaled

def create_negative_examples():
    max_neg_per_scale = 2  # Vrem 5 imagini negative per scară.
    annotations_dict = load_annotations()
    all_image_paths = list(annotations_dict.keys())
    total_neg_created = 0

    # all_image_paths = all_image_paths[0:1]
    for image_path in all_image_paths:
        img = cv.imread(image_path)
        if img is None:
            logging.warning(f"Could not load image: {image_path}. Skipping.")
            continue

        pyramid, scales = create_pyramid(img, scale_factor=0.5, min_size=(49, 49))
        pos_boxes_original = annotations_dict[image_path]

        # Parcurgem fiecare scară din piramidă
        for idx, resized_img in enumerate(pyramid):
            scale_neg_count = 0  # contor pentru negative examples la scara curentă
            scaled_pos_boxes = scale_bboxes(pos_boxes_original, scales[idx])
            height, width = resized_img.shape[:2]

            # Începem să generăm patch-uri negative la scara curentă
            # Încercăm un număr mare de tentative, dar ne oprim dacă am extras 5
            for _ in range(50):  
                if scale_neg_count >= max_neg_per_scale:
                    break

                w, h = 49, 49  # exemplu; poți pune logică random
                if w > width or h > height:
                    continue

                x_min = random.randint(0, width - w)
                y_min = random.randint(0, height - h)
                x_max = x_min + w
                y_max = y_min + h

                candidate_box = (x_min, y_min, x_max, y_max)
                
                # Verifică să nu se suprapună prea mult cu bounding box-urile scale
                if not does_intersect_with_any(scaled_pos_boxes, candidate_box, max_iou_allowed=0.05):
                    cropped = resized_img[y_min:y_max, x_min:x_max]
                    if cropped.size == 0:
                        continue

                    neg_name = f"{total_neg_created:05d}.jpg"
                    out_path = os.path.join(negative_folder, neg_name)
                    cv.imwrite(out_path, cropped)

                    scale_neg_count += 1
                    total_neg_created += 1
                    logging.info(
                        f"Saved negative example: {out_path} | scale={scales[idx]:.3f} | pyramid_index={idx}"
                    )

        logging.info(f"Processed image: {image_path}")

    logging.info(f"All done. Total negative examples created: {total_neg_created}")

if __name__ == '__main__':
    create_negative_examples()
