import os
import cv2 as cv
import random

class NegativeSamplesGenerator:
    def __init__(self, params):
        self.params = params
        self.characters = params.characters
        self.aspect_ratios = params.aspect_ratios
        self.negative_folder = os.path.join(params.base_dir, 'negative')

        self.ratio_distribution = {ar: 0 for ar in self.aspect_ratios}

        if not os.path.exists(self.negative_folder):
            os.makedirs(self.negative_folder)
            print(f"Created negative folder: {self.negative_folder}")

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
        return cv.resize(cropped_image, (new_w, new_h), interpolation=cv.INTER_AREA)

    def patch_size(self, width_img, height_img):
        chosen_ratio = random.choice(self.aspect_ratios)

        if chosen_ratio >= 1:
            h_patch = self.params.dim_window
            w_patch = int(round(chosen_ratio * h_patch))
        else:
            w_patch = self.params.dim_window
            h_patch = int(round(w_patch / chosen_ratio))

        if w_patch > width_img or h_patch > height_img:
            return None

        return (w_patch, h_patch, chosen_ratio)

    def iou(self, boxA, boxB):
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

    def does_intersect_with_any(self, pos_boxes, neg_box, max_iou_allowed=0.0):
        for pb in pos_boxes:
            if self.iou(pb, neg_box) > max_iou_allowed:
                return True
        return False

    def load_annotations(self):
        annotations_dict = {}
        for character in self.characters:
            annotation_file = os.path.join(self.params.input_dir, f'{character}_annotations.txt')
            characters_images_folder = os.path.join(self.params.input_dir, character)
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

    def create_pyramid(self, img):
        pyramid, scales = [], []
        current_img = img.copy()
        current_scale = 1.0
        while current_img.shape[0] >= self.params.dim_window and current_img.shape[1] >=self.params.dim_window and current_scale >= 0.3:
            pyramid.append(current_img)
            scales.append(current_scale)
            new_w = int(current_img.shape[1] * self.params.scale)
            new_h = int(current_img.shape[0] * self.params.scale)
            current_scale *= self.params.scale
            current_img = cv.resize(img, (new_w, new_h))
        return pyramid, scales

    def scale_bboxes(self, bboxes, scale):
        scaled = []
        for (x_min, y_min, x_max, y_max) in bboxes:
            scaled.append((
                int(x_min * scale),
                int(y_min * scale),
                int(x_max * scale),
                int(y_max * scale)
            ))
        return scaled

    def create_negative_examples(self):
        max_neg_per_scale = 4
        annotations_dict = self.load_annotations()
        all_image_paths = list(annotations_dict.keys())
        total_neg_created = 0

        for image_path in all_image_paths:
            img = cv.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}. Skipping.")
                continue

            pyramid, scales = self.create_pyramid(img)
            pos_boxes_original = annotations_dict[image_path]

            for idx, resized_img in enumerate(pyramid):
                scale_neg_count = 0
                scaled_pos_boxes = self.scale_bboxes(pos_boxes_original, scales[idx])
                height, width = resized_img.shape[:2]

                for _ in range(50):
                    if scale_neg_count >= max_neg_per_scale:
                        break

                    result = self.patch_size(width, height)
                    if result is None:
                        continue

                    w_patch, h_patch, chosen_ratio = result

                    x_min = random.randint(0, width - w_patch)
                    y_min = random.randint(0, height - h_patch)
                    x_max = x_min + w_patch
                    y_max = y_min + h_patch

                    candidate_box = (x_min, y_min, x_max, y_max)

                    if not self.does_intersect_with_any(scaled_pos_boxes, candidate_box, max_iou_allowed=0.05):
                        cropped = resized_img[y_min:y_max, x_min:x_max]
                        if cropped.size == 0:
                            continue

                        resized_cropped = self.resize_with_aspect_ratio(cropped, chosen_ratio)

                        ratio_subfolder = os.path.join(self.negative_folder, self.ratio_to_folder(chosen_ratio))
                        if not os.path.exists(ratio_subfolder):
                            os.makedirs(ratio_subfolder)

                        neg_name = f"{total_neg_created:05d}.jpg"
                        out_path = os.path.join(ratio_subfolder, neg_name)
                        cv.imwrite(out_path, resized_cropped)

                        scale_neg_count += 1
                        total_neg_created += 1
                        print(f"Saved negative example: {out_path} | scale={scales[idx]:.3f} | pyramid_index={idx}")

            print(f"Processed image: {image_path}")

        print(f"All done. Total negative examples created: {total_neg_created}")

        print("Distribuția pe aspect ratios (negative):")
        for ar in self.aspect_ratios:
            print(f"  ratio={ar}: {self.ratio_distribution[ar]} patch-uri")