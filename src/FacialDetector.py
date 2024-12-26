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
import datetime

class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_models = {}

    def ratio_to_folder(self, r):
        ratio_str = str(r).replace('.', '')
        return f"ratio_{ratio_str}"

    def check_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Directorul a fost creat: {folder}")
        else:
            print(f"Directorul {folder} exista")

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
            if img is None:
                print(f"Imaginea {file_path} nu a fost încărcată corect.")
                continue
            combined_features = self.compute_descriptors(img)
            descriptors_list.append(combined_features)

            if self.params.use_flip_images:
                flipped_img = np.fliplr(img)
                combined_features = self.compute_descriptors(flipped_img)
                descriptors_list.append(combined_features)

        descriptors_list = np.array(descriptors_list)
        return descriptors_list

    def get_positive_descriptors_for_ratio(self, r):
        ratio_folder = self.ratio_to_folder(r)
        samples_pos_dir = os.path.join(self.params.dir_pos_examples, ratio_folder)

        metrics_pos_dir = os.path.join(self.params.dir_save_files, ratio_folder)
        positive_features_path = os.path.join(metrics_pos_dir, 'pozitive-descriptors_' + str(self.params.dim_window) + '_' + str(self.params.dim_hog_cell) + '_' + str(self.params.bins) + '.npy')

        if os.path.exists(positive_features_path):
            print('Am incarcat descriptorii pentru exemplele pozitive')
        else:
            print('Construim descriptorii pentru exemplele pozitive:')
            positive_features = self.get_descriptors_from_directory(samples_pos_dir, label="pozitive")
            self.check_folder(metrics_pos_dir)

            np.save(positive_features_path, positive_features)

            # TODO 
            self.pos_count = len(positive_features)

    def get_negative_descriptors_for_ratio(self, r):
        ratio_folder = self.ratio_to_folder(r)

        neg_samples_dir = os.path.join(self.params.dir_neg_examples, ratio_folder)
        hard_neg_samples_dir = os.path.join(self.params.dir_hard_neg_examples, ratio_folder)
        metrics_dir = os.path.join(self.params.dir_save_files, ratio_folder)
        negative_features_path = os.path.join(metrics_dir, 'negative-descriptors_' + str(self.params.dim_window) + '_' + str(self.params.dim_hog_cell) + '_' + str(self.params.bins) + '.npy')

        if os.path.exists(negative_features_path):
            print('Am incarcat descriptorii pentru exemplele negative + hard negative')
            all_negative_examples = np.load(negative_features_path)
        else:
            print('Construim descriptorii pentru exemplele negative:')
            negative_features = self.get_descriptors_from_directory(neg_samples_dir, label="negative")

            if self.params.use_hard_mining:
                print('Construim descriptorii pentru exemplele hard negative:')
                hard_negative_features = self.get_descriptors_from_directory(hard_neg_samples_dir, label="hard-negative")
            else:
                hard_negative_features = np.array([])

            if hasattr(self, 'pos_count'):
                neg_count = self.pos_count * 3 

                normal_neg_count = int(neg_count * 0.7)
                hard_neg_count = neg_count - normal_neg_count

                print(f"-> Avem {self.pos_count} pozitive.")
                print(f"-> Dorim in total ~{neg_count} negative (70% din set).")
                print(f"    - {normal_neg_count} negative normale")
                print(f"    - {hard_neg_count} hard negative")
            else:
                normal_neg_count = len(negative_features)
                hard_neg_count = len(hard_negative_features)

            all_negatives = negative_features
            if len(hard_negative_features) > 0:
                print(f"am folosit {len(hard_negative_features)} hard negative.")
                all_negatives = np.concatenate((negative_features, hard_negative_features), axis=0)

            total_available = len(all_negatives)
            print(f"Total negative disponibile (normale + hard): {total_available}")

            desired_count = normal_neg_count + hard_neg_count
            if total_available > desired_count:
                print(f"Vom face random sampling la {desired_count} din {total_available} negative totale.")
                indices = np.arange(total_available)
                np.random.shuffle(indices)
                selected_indices = indices[:desired_count]
                all_negative_examples = all_negatives[selected_indices]
            else:
                print("Numărul total de negative este mai mic sau egal cu necesarul, le luăm pe toate.")
                all_negative_examples = all_negatives

            self.check_folder(metrics_dir)
            np.save(negative_features_path, all_negative_examples)

        print(f"Numar total de exemple negative incarcate (final): {len(all_negative_examples)}")
        return all_negative_examples
 

    def train_classifier(self, r):
        ratio_folder = self.ratio_to_folder(r)

        pos_descriptor_path = os.path.join(self.params.dir_save_files, ratio_folder, 'pozitive-descriptors_'  + str(self.params.dim_window) + '_' + str(self.params.dim_hog_cell) + '_' + str(self.params.bins) + '.npy')
        neg_descriptor_path = os.path.join(self.params.dir_save_files, ratio_folder, 'negative-descriptors_'  + str(self.params.dim_window) + '_' + str(self.params.dim_hog_cell) + '_' + str(self.params.bins) + '.npy')

        positive_features = np.load(pos_descriptor_path)
        negative_features = np.load(neg_descriptor_path)

        print("Positive features shape:", positive_features.shape)
        print("Negative features shape:", negative_features.shape)
        training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
        train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))

        
        svm_file_name = os.path.join(
            self.params.dir_save_files,
            ratio_folder,
            'best_model_%d_%d_%d_%d_%d' % (
                self.params.dim_window,
                self.params.dim_hog_cell,
                self.params.bins,
                self.params.number_negative_examples,
                self.params.number_positive_examples
            )
        )
        if os.path.exists(svm_file_name):
            self.best_models[r] = pickle.load(open(svm_file_name, 'rb'))
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
        self.best_models[r] = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        # plt.plot(np.sort(positive_scores))
        # plt.plot(np.zeros(len(positive_scores)))
        # plt.plot(np.sort(negative_scores))
        # plt.xlabel('Nr example antrenare')
        # plt.ylabel('Scor clasificator')
        # plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        # plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        # plt.show()

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

    def create_gaussian_pyramid_with_scales(self, img):
        pyramid = [img]
        scales = [1.0]
        
        while img.shape[0] > self.params.dim_window * 0.8 and img.shape[1] > self.params.dim_window * 0.8 and scales[-1] * self.params.scale >= 0.3:
            new_width = int(img.shape[1] * self.params.scale)
            new_height = int(img.shape[0] * self.params.scale)
            
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
            pyramid.append(img)
            scales.append(scales[-1] * self.params.scale)
        
        return pyramid, scales

    def collect_hard_negatives(self):
        for r in self.params.aspect_ratios:

            if r not in self.best_models:
                print(f"[HARD MINING] Nu există un model pentru aspect ratio-ul {r}. Sari peste.")
                continue

            print(f"[HARD MINING] Începem colectarea pentru aspect ratio={r} ...")

            ratio_folder = self.ratio_to_folder(r)  # ex: "ratio_15" dacă r=1.5
            hard_neg_subfolder = os.path.join(self.params.dir_hard_neg_examples, ratio_folder)
            if not os.path.exists(hard_neg_subfolder):
                os.makedirs(hard_neg_subfolder)

            model = self.best_models[r]
            w = model.coef_.T
            bias = model.intercept_[0]

            counter = 0

            if r > 1.0:
                h_patch = self.params.dim_window
                w_patch = int(round(r * h_patch))
            else:
                w_patch = self.params.dim_window
                h_patch = int(round(w_patch / r))

            for ch in self.params.characters:
                if counter > 2500:
                    break
                print(f"[HARD MINING - {r}] Procesăm caracterul {ch} ...")
                ground_truth_dict = {}
                with open(os.path.join(self.params.train_adnotations, ch + '_annotations.txt'), 'r') as f:
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

                train_images_path = os.path.join(self.params.dir_train_examples, ch, '*.jpg')
                train_files = glob.glob(train_images_path)

                hard_negative_descriptors = []

                for i, img_path in enumerate(train_files):
                    short_name = ntpath.basename(img_path)
                    print(f'[HARD MINING - {r}] Procesăm {short_name} ({i+1}/{len(train_files)})')
                    img = cv.imread(img_path)
                    if img is None:
                        continue

                    # Cream piramida în funcție de dimensiunea ferestrei (h_patch, w_patch)
                    pyramid, scales = self.create_gaussian_pyramid_with_scales(img)

                    image_detections = []
                    image_scores = []

                    # Calculăm numărul de celule HOG pe orizontală/verticală
                    num_cell_in_w = (w_patch // self.params.dim_hog_cell) - 1
                    num_cell_in_h = (h_patch // self.params.dim_hog_cell) - 1

                    for scale_idx, resized_img in enumerate(pyramid):
                        current_scale = scales[scale_idx]

                        num_cols = resized_img.shape[1] // self.params.dim_hog_cell - 1
                        num_rows = resized_img.shape[0] // self.params.dim_hog_cell - 1

                        for y in range(0, num_rows - num_cell_in_h):
                            for x in range(0, num_cols - num_cell_in_w):
                                x_min_local = x * self.params.dim_hog_cell
                                y_min_local = y * self.params.dim_hog_cell
                                x_max_local = x_min_local + w_patch
                                y_max_local = y_min_local + h_patch

                                window = resized_img[y_min_local:y_max_local, x_min_local:x_max_local]
                                descr = self.compute_descriptors(window)
                                score = np.dot(descr, w)[0] + bias

                                if score > 2.5:
                                    # Transformăm coordonatele la scara originală
                                    x_min = int(x_min_local / current_scale)
                                    y_min = int(y_min_local / current_scale)
                                    x_max = int(x_max_local / current_scale)
                                    y_max = int(y_max_local / current_scale)
                                    image_detections.append([x_min, y_min, x_max, y_max])
                                    image_scores.append(score)

                    # NMS
                    if len(image_scores) > 0:
                        image_detections, image_scores = self.non_maximal_suppression(
                            np.array(image_detections),
                            np.array(image_scores), img.shape
                        )
                        image_detections = image_detections.tolist()
                        image_scores = image_scores.tolist()

                    gt_bboxes = ground_truth_dict.get(short_name, [])

                    for det_idx, bbox_det in enumerate(image_detections):
                        x1_det, y1_det, x2_det, y2_det = bbox_det
                        ious = [self.intersection_over_union(bbox_det, gt) for gt in gt_bboxes]
                        max_iou = max(ious) if len(ious) > 0 else 0

                        if max_iou < 0.25:
                            patch = img[y1_det:y2_det, x1_det:x2_det]

                            patch_name = f"{image_scores[det_idx]:.3f}_{counter:05d}.jpg"
                            patch_path = os.path.join(hard_neg_subfolder, patch_name)

                            patch = cv.resize(patch, (w_patch, h_patch))
                            cv.imwrite(patch_path, patch)

                            counter += 1

                if len(hard_negative_descriptors) == 0:
                    print(f"[{r}] Nu am găsit hard negatives cu scor > 1.0 și IoU=0!")
                    continue
    
    def run(self):
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None
        scores = np.array([])
        num_test_images = len(test_files)
        file_names = np.array([])

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i + 1, num_test_images))
            img = cv.imread(test_files[i])

            image_scores = []
            image_detections = []

            for r in self.params.aspect_ratios:        
                # Verificare dacă există model antrenat pentru aspect ratio-ul curent
                if r not in self.best_models:
                    print(f"Nu am gasit modelul pentru aspect ratio-ul {r}")
                    continue
                else:
                    self.best_model = self.best_models[r]
                    w = self.best_model.coef_.T
                    bias = self.best_model.intercept_[0]

                # Calculăm înălțimea și lățimea ferestrei pe baza aspect ratio-ului
                if r > 1:
                    # r = w_patch / h_patch  =>  w_patch = r * h_patch
                    h_patch = self.params.dim_window
                    w_patch = int(round(r * h_patch))
                else:
                    # r = w_patch / h_patch  =>  h_patch = w_patch / r
                    w_patch = self.params.dim_window
                    h_patch = int(round(w_patch / r))

                pyramid, scales = self.create_gaussian_pyramid_with_scales(img)


                # => calculăm numărul de celule pe orizontală și verticală
                #    de ex: (w_patch // dim_hog_cell) - 1 și (h_patch // dim_hog_cell) - 1
                num_cell_in_w = (w_patch // self.params.dim_hog_cell) - 1
                num_cell_in_h = (h_patch // self.params.dim_hog_cell) - 1

                for j, resized_img in enumerate(pyramid):
                    current_scale = scales[j]

                    # num_cols și num_rows se referă la numărul total de celule orizontale/verticale
                    num_cols = resized_img.shape[1] // self.params.dim_hog_cell - 1
                    num_rows = resized_img.shape[0] // self.params.dim_hog_cell - 1

                    for y in range(0, num_rows - num_cell_in_h):
                        for x in range(0, num_cols - num_cell_in_w):
                            x_min_local = x * self.params.dim_hog_cell
                            y_min_local = y * self.params.dim_hog_cell
                            x_max_local = x_min_local + w_patch
                            y_max_local = y_min_local + h_patch

                            window = resized_img[y_min_local:y_max_local, x_min_local:x_max_local]

                            # Extragem descriptorul (e.g. HOG) și calculăm scorul
                            descr = self.compute_descriptors(window)
                            score = np.dot(descr, w)[0] + bias
                            if score > self.params.threshold:
                                print(score)

                                # Convertim coordonatele înapoi la dimensiunea originală
                                x_min = int(x_min_local / current_scale)
                                y_min = int(y_min_local / current_scale)
                                x_max = int(x_max_local / current_scale)
                                y_max = int(y_max_local / current_scale)

                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)

                # Non-maximal suppression pentru a elimina suprapunerile
            if len(image_scores) > 0:
                print("Aplicăm NMS...")
                print(image_scores)
                image_detections, image_scores = self.non_maximal_suppression(
                    np.array(image_detections),
                    np.array(image_scores), img.shape
                )
                print("Am aplicat NMS")
                print(image_scores)
                image_detections = image_detections.tolist()
                image_scores = image_scores.tolist()

            # Salvăm detecțiile pentru imaginea curentă
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
        print('aici')
        print(num_gt_detections)
        print(cum_true_positive)
        print(cum_false_positive)
        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)

        print(f"True Positives: {np.sum(true_positive)}")
        print(f"False Positives: {np.sum(false_positive)}")
        print(f"Recall: {np.average(rec)}")
        print(f"Precision: {np.average(prec)}")

        fp_value = len(false_positive)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f, FP: %d' % (average_precision, fp_value))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'precizie_medie_{timestamp}_WINDOW_{self.params.dim_window}_FP_{fp_value}.png'

        plt.savefig(os.path.join(self.params.dir_save_files, file_name))
        plt.show()