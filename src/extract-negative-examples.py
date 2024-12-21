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

    iou_value = interArea / float(boxAArea + boxBArea - interArea)
    return iou_value

# subject of discussion bcs I could leave a small error here like 5%
def does_intersect_with_any(pos_boxes, neg_box):
    for pb in pos_boxes:
        if iou(pb, neg_box) > 0:
            return True
    return False

# 
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

def create_negative_examples(max_neg_per_image=10):
    # these are some of the aspect ratio that I did find by using count-aspect-ratios.py
    # now of course i did select only the top 5 and compute manually the probabilities
    # aspect_ratios = [1.0, 1.2, 1.15, 1.25, 1.3]
    # probabilities = [0.815431, 0.049924, 0.045386, 0.045386, 0.043873]

    aspect_ratios = [1.0]
    probabilities = [1.0]

    annotations_dict = load_annotations() 
    all_image_paths = list(annotations_dict.keys())
    
    total_neg_created = 0

    for image_path in all_image_paths:
        img = cv.imread(image_path)
        if img is None:
            logging.warning(f"Could not load image: {image_path}. Skipping.")
            continue

        height, width = img.shape[:2]
        pos_boxes = annotations_dict[image_path]

        neg_created_for_this_image = 0

        while neg_created_for_this_image < max_neg_per_image:
            valid_patch_found = False

            # I did assuma that some of images don't have enough negative examples
            for _ in range(5):
                ratio = random.choices(aspect_ratios, weights=probabilities, k=1)[0]
                # w = random.randint(30, 80)  # random width
                w = 49 # just temp
                h = int(round(w * ratio)) # and based on that random width we compute the height

                if w > width or h > height:
                    continue

                x_min = random.randint(0, width - w)
                y_min = random.randint(0, height - h)
                x_max = x_min + w
                y_max = y_min + h

                candidate_box = (x_min, y_min, x_max, y_max)

                if not does_intersect_with_any(pos_boxes, candidate_box):
                    cropped = img[y_min:y_max, x_min:x_max]
                    if cropped.size == 0:
                        continue

                    base_name = os.path.basename(image_path)
                    file_stem = os.path.splitext(base_name)[0]
                    neg_name = (
                        f"{total_neg_created:05d}.jpg"
                    )
                    out_path = os.path.join(negative_folder, neg_name)
                    cv.imwrite(out_path, cropped)

                    valid_patch_found = True
                    neg_created_for_this_image += 1
                    total_neg_created += 1

                    logging.info(f"Saved negative example: {out_path}")
                    break

            if not valid_patch_found:
                # after 5 attempts we should stop trying to find a valid patch
                logging.info(f"Could not find more negative patches in {image_path}. Moving on.")
                break

    logging.info(f"All done. Total negative examples created: {total_neg_created}")

if __name__ == '__main__':
    create_negative_examples(max_neg_per_image=10)