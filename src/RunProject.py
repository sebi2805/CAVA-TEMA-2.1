from Parameters import *
from FacialDetector import *
from PositiveSamplesGenerator import *
from NegativeSamplesGenerator import *
import os
from Visualize import *


params: Parameters = Parameters()


params.use_hard_mining = False
params.use_flip_images = True
params.has_annotations = True

if params.use_flip_images:
    params.number_positive_examples *= 2

if not os.path.exists(params.dir_pos_examples) or len(os.listdir(params.dir_pos_examples)) == 0:
    PositiveSamplesGenerator(params).create_pozitive_samples()

if not os.path.exists(params.dir_neg_examples) or len(os.listdir(params.dir_neg_examples)) == 0:
    NegativeSamplesGenerator(params).create_negative_examples()

facial_detector: FacialDetector = FacialDetector(params)

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente
for r in params.aspect_ratios:
    facial_detector.get_positive_descriptors_for_ratio(r)
    facial_detector.get_negative_descriptors_for_ratio(r)

    facial_detector.train_classifier(r)

# facial_detector.collect_hard_negatives()

detections, scores, file_names = facial_detector.run()

if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:
    show_detections_without_ground_truth(detections, scores, file_names, params)

if not os.path.exists(params.task1_solution_dir):
    os.makedirs(params.task1_solution_dir)

np.save(os.path.join(params.task1_solution_dir, 'detections_all_faces.npy'), detections)
np.save(os.path.join(params.task1_solution_dir, 'scores_all_faces.npy'), scores)
np.save(os.path.join(params.task1_solution_dir, 'file_names_all_faces.npy'), file_names)