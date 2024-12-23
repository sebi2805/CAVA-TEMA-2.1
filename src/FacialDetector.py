from Parameters import *
import os
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog

class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None

    def compute_descriptors(self, img):
        hog_features = hog(
            cv.cvtColor(img, cv.COLOR_BGR2GRAY),
            pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
            cells_per_block=(2, 2), feature_vector=True
        )

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hist = cv.calcHist(
            [hsv], [0, 1, 2], None,
            [self.params.bins, self.params.bins, self.params.bins],
            [0, 180, 0, 256, 0, 256]
        )

        color_features = cv.normalize(hist, hist).flatten()

        return np.concatenate((hog_features, color_features))

    def get_descriptors_from_directory(self, directory, label):
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    files.append(os.path.join(root, filename))

        num_images = len(files)
        descriptors_list = []
        print(f"Calculam descriptorii pt {num_images} imagini {label}...")

        for i, file_path in enumerate(files):
            print(f"Procesam exemplul {label} numarul {i}...")
            img = cv.imread(file_path)

            combined_features = self.compute_descriptors(img)
            descriptors_list.append(combined_features)

            if self.params.use_flip_images:
                flipped_img = np.fliplr(img)
                combined_features = self.compute_descriptors(flipped_img)
                descriptors_list.append(combined_features)

        descriptors_list = np.array(descriptors_list)
        return descriptors_list

    def get_positive_descriptors(self):
        return self.get_descriptors_from_directory(self.params.dir_pos_examples, label="pozitive")

    def get_negative_descriptors(self):
        negative_examples = self.get_descriptors_from_directory(self.params.dir_neg_examples, label="negative")
        hard_negative_examples = self.get_descriptors_from_directory(self.params.dir_hard_neg_examples, label="hard-negative")

        if len(hard_negative_examples) > 0:
            negative_examples = np.concatenate((negative_examples, hard_negative_examples))
        
        return negative_examples

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(
            self.params.dir_save_files,
            'best_model_%d_%d_%d' % (
                self.params.dim_hog_cell,
                self.params.number_negative_examples,
                self.params.number_positive_examples
            )
        )
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:
                        if self.intersection_over_union(sorted_image_detections[i], sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                               sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def create_gaussian_pyramid_with_scales(self, img, scale_factor=0.5, min_size=(64, 64)):
        pyramid = [img]
        scales = [1.0]
        while img.shape[0] > min_size[0] and img.shape[1] > min_size[1]:
            img = cv.pyrDown(img)
            pyramid.append(img)
            scales.append(scales[-1] * scale_factor)
        return pyramid, scales

    def collect_hard_negatives(self):
        ground_truth_dict = {}
        with open(self.params.train_adnotations, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                file_name = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:5])
                if file_name not in ground_truth_dict:
                    ground_truth_dict[file_name] = []
                ground_truth_dict[file_name].append([x1, y1, x2, y2])

        train_images_path = os.path.join(self.params.dir_train_examples, '*.jpg')
        train_files = glob.glob(train_images_path)

        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]

        hard_negative_descriptors = []

        for i, img_path in enumerate(train_files):
            short_name = ntpath.basename(img_path)
            print(f'[HARD MINING] Procesăm {short_name} ({i+1}/{len(train_files)})')
            img = cv.imread(img_path)
            if img is None:
                continue

            pyramid, scales = self.create_gaussian_pyramid_with_scales(
                img, scale_factor=0.5, min_size=(self.params.dim_window, self.params.dim_window)
            )

            image_detections = []
            image_scores = []

            for scale_idx, resized_img in enumerate(pyramid):
                current_scale = scales[scale_idx]
                num_cols = resized_img.shape[1] // self.params.dim_hog_cell - 1
                num_rows = resized_img.shape[0] // self.params.dim_hog_cell - 1
                num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(0, num_rows - num_cell_in_template):
                    for x in range(0, num_cols - num_cell_in_template):
                        x_min_local = x * self.params.dim_hog_cell
                        y_min_local = y * self.params.dim_hog_cell
                        x_max_local = x_min_local + self.params.dim_window
                        y_max_local = y_min_local + self.params.dim_window

                        window = resized_img[y_min_local:y_max_local, x_min_local:x_max_local]
                        descr = self.compute_descriptors(window)
                        score = np.dot(descr, w)[0] + bias

                        if score > 1.0:
                            x_min = int(x_min_local / current_scale)
                            y_min = int(y_min_local / current_scale)
                            x_max = int(x_max_local / current_scale)
                            y_max = int(y_max_local / current_scale)
                            image_detections.append([x_min, y_min, x_max, y_max])
                            image_scores.append(score)

            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(
                    np.array(image_detections),
                    np.array(image_scores), img.shape
                )
                image_detections = image_detections.tolist()
                image_scores = image_scores.tolist()

            if short_name in ground_truth_dict:
                gt_bboxes = ground_truth_dict[short_name]
            else:
                gt_bboxes = []

            for det_idx, bbox_det in enumerate(image_detections):
                x1_det, y1_det, x2_det, y2_det = bbox_det
                score_det = image_scores[det_idx]

                ious = [self.intersection_over_union(bbox_det, gt) for gt in gt_bboxes]
                max_iou = max(ious) if len(ious) > 0 else 0

                if max_iou == 0:
                    patch = img[y1_det:y2_det, x1_det:x2_det]
                    cv.imwrite(f'{self.params.dir_hard_neg_examples}/{(len(hard_negative_descriptors) + 3957):04d}.jpg', patch)
                    if patch.shape[0] == self.params.dim_window and patch.shape[1] == self.params.dim_window:
                        patch_descr = self.compute_descriptors(patch)
                        hard_negative_descriptors.append(patch_descr)

        if len(hard_negative_descriptors) == 0:
            print("Nu am găsit  hard negatives cu scor > 3.0 și IoU=0!")
            return np.array([])

        hard_negative_descriptors = np.array(hard_negative_descriptors)
        print(f"Am colectat {len(hard_negative_descriptors)} hard negatives.")
        return hard_negative_descriptors
    
    def run(self):
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None
        scores = np.array([])
        file_names = np.array([])
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i + 1, num_test_images))
            img = cv.imread(test_files[i])

            scale = 0.5
            min_size = (self.params.dim_window, self.params.dim_window)
            pyramid, scales = self.create_gaussian_pyramid_with_scales(img, scale, min_size)

            image_scores = []
            image_detections = []

            for j, resized_img in enumerate(pyramid):
                current_scale = scales[j]
                num_cols = resized_img.shape[1] // self.params.dim_hog_cell - 1
                num_rows = resized_img.shape[0] // self.params.dim_hog_cell - 1
                num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(0, num_rows - num_cell_in_template):
                    for x in range(0, num_cols - num_cell_in_template):
                        x_min_local = x * self.params.dim_hog_cell
                        y_min_local = y * self.params.dim_hog_cell
                        x_max_local = x_min_local + self.params.dim_window
                        y_max_local = y_min_local + self.params.dim_window

                        window = resized_img[y_min_local:y_max_local, x_min_local:x_max_local]

                        descr = self.compute_descriptors(window)
                        score = np.dot(descr, w)[0] + bias

                        if score > self.params.threshold:
                            x_min = int(x_min_local / current_scale)
                            y_min = int(y_min_local / current_scale)
                            x_max = int(x_max_local / current_scale)
                            y_max = int(y_max_local / current_scale)
                            image_detections.append([x_min, y_min, x_max, y_max])
                            image_scores.append(score)

            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(
                    np.array(image_detections),
                    np.array(image_scores), img.shape
                )
                image_detections = image_detections.tolist()
                image_scores = image_scores.tolist()

            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))
                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesare al imaginii de testare %d/%d este %f sec.'
                  % (i + 1, num_test_images, end_time - start_time))

        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
