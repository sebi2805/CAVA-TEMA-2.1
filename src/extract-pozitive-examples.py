import os
import cv2 as cv
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_folder = 'antrenare'
output_folder = 'output/pozitive'
characters = ['dad', 'mom', 'dexter', 'deedee']

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    logging.info(f"Created output folder: {output_folder}")

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h),
                            flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_REPLICATE)
    return rotated

def create_pozitive_samples():
    for character in characters:
        characters_images = os.path.join(input_folder, character)
        annotation_file = os.path.join(input_folder, f'{character}_annotations.txt')

        if not os.path.exists(characters_images):
            logging.error(f"Character images folder does not exist: {characters_images}")
            continue

        if not os.path.exists(annotation_file):
            logging.error(f"Annotation file does not exist: {annotation_file}")
            continue

        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                image_path = os.path.join(characters_images, line[0])

                img = cv.imread(image_path)
                if img is None:
                    logging.warning(f"Failed to load image: {image_path}")
                    continue

                x_min, y_min, x_max, y_max = map(int, line[1:5])
                chr = line[5]

                cropped_image = img[y_min:y_max, x_min:x_max]

                character_output_folder = os.path.join(output_folder, chr)
                if not os.path.exists(character_output_folder):
                    os.makedirs(character_output_folder)
                    logging.info(f"Created character folder: {character_output_folder}")

                # Salvează imaginea decupată
                output_filename = f"{character}_{chr}_{line[0]}"
                output_path = os.path.join(character_output_folder, output_filename)
                cv.imwrite(output_path, cropped_image)
                logging.info(f"Cropped image saved: {output_path}")

                # Facem două rotații: una la stânga, una la dreapta
                angles = [-random.randint(10, 30), random.randint(10, 30)]
                for r, angle in enumerate(angles, start=1):
                    rotated_cropped = rotate_image(cropped_image, angle)
                    rotated_filename = f"{character}_{chr}_{line[0].split('.')[0]}_rot{r}.jpg"
                    rotated_path = os.path.join(character_output_folder, rotated_filename)
                    cv.imwrite(rotated_path, rotated_cropped)
                    logging.info(f"Rotated image (angle={angle}) saved: {rotated_path}")

if __name__ == '__main__':
    create_pozitive_samples()
