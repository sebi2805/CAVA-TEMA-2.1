import os
import cv2 as cv
import random
import math

class PositiveSamplesGenerator:
    def __init__(self, params):
        self.params = params
        
        self.input_folder = params.input_dir
        self.output_folder = params.dir_pos_examples
        self.characters = params.characters
        self.aspect_ratios = params.aspect_ratios
        # Dicționar pentru a contoriza distribuția fiecărui aspect ratio folosit
        self.ratio_distribution = {ar: 0 for ar in self.aspect_ratios}

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder: {self.output_folder}")

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(
            image,
            M,
            (w, h),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_REPLICATE
        )
        return rotated

    def get_closest_ratio(self, ratio):
        return min(self.aspect_ratios, key=lambda x: abs(x - ratio))

    def ratio_to_folder(self, r):
        ratio_str = str(r).replace('.', '')
        return f"ratio_{ratio_str}"

    def resize_with_aspect_ratio(self, cropped_image, closest_ratio):
        h, w = cropped_image.shape[:2]
        if closest_ratio >= 1:
            new_h = self.params.dim_window
            new_w = int(round(new_h * closest_ratio))
        else:
            new_w = self.params.dim_window
            new_h = int(round(new_w / closest_ratio))

        resized_cropped = cv.resize(cropped_image, (new_w, new_h), interpolation=cv.INTER_AREA)
        return resized_cropped

    def create_pozitive_samples(self):
        for character in self.characters:
            characters_images = os.path.join(self.input_folder, character)
            annotation_file = os.path.join(self.input_folder, f'{character}_annotations.txt')

            if not os.path.exists(characters_images):
                print(f"Character images folder does not exist: {characters_images}")
                continue

            if not os.path.exists(annotation_file):
                print(f"Annotation file does not exist: {annotation_file}")
                continue

            with open(annotation_file, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    image_path = os.path.join(characters_images, line[0])

                    img = cv.imread(image_path)
                    if img is None:
                        print(f"Failed to load image: {image_path}")
                        continue

                    x_min, y_min, x_max, y_max = map(int, line[1:5])
                    cropped_image = img[y_min:y_max, x_min:x_max]
                    h, w = cropped_image.shape[:2]

                    if h <= 0 or w <= 0:
                        print(f"Skipping invalid crop: {image_path}")
                        continue

                    actual_ratio = w / float(h)
                    closest_ratio = self.get_closest_ratio(actual_ratio)
                    self.ratio_distribution[closest_ratio] += 1

                    resized_cropped = self.resize_with_aspect_ratio(cropped_image, closest_ratio)

                    ratio_subfolder = os.path.join(self.output_folder, self.ratio_to_folder(closest_ratio))
                    if not os.path.exists(ratio_subfolder):
                        os.makedirs(ratio_subfolder)
                        print(f"Created ratio subfolder: {ratio_subfolder}")

                    output_filename = f"{character}_{line[0]}"
                    output_path = os.path.join(ratio_subfolder, output_filename)
                    cv.imwrite(output_path, resized_cropped)
                    print(f"Cropped+Resized image saved: {output_path}")

                    angles = [22, 45]
                    for r, angle in enumerate(angles, start=1):
                        rotated_cropped = self.rotate_image(resized_cropped, angle)
                        rotated_filename = f"{character}_{line[0].split('.')[0]}_rot{r}.jpg"
                        rotated_path = os.path.join(ratio_subfolder, rotated_filename)
                        cv.imwrite(rotated_path, rotated_cropped)
                        print(f"Rotated image (angle={angle}) saved: {rotated_path}")

        print("Distribuția pe aspect ratios:")
        for r in self.aspect_ratios:
            print(f"  ratio={r}: {self.ratio_distribution[r]} ferestre")