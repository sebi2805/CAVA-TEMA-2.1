import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Paths to the .npy files
detections_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\yolo\task1\detections_all_faces.npy"
filenames_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\yolo\task1\file_names_all_faces.npy"
scores_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\yolo\task1\scores_all_faces.npy"

# Folder containing the validation images
images_folder = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\validare\validare"

# Load the .npy files
detections = np.load(detections_path, allow_pickle=True)  # (number_of_images, variable_per_image_boxes)
filenames = np.load(filenames_path, allow_pickle=True)    # (number_of_images,)
scores = np.load(scores_path, allow_pickle=True)          # (number_of_images, variable_per_image_scores)

# Function to display images with bounding boxes and scores
def display_images_with_detections():
    for filename, boxes, score_list in zip(filenames, detections, scores):
        img_path = os.path.join(images_folder, filename)
        if not os.path.exists(img_path):
            print(f"Image {filename} not found!")
            continue

        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {filename}")
            continue

        # Check if boxes and scores are arrays or single values
        if isinstance(boxes, np.ndarray) and boxes.ndim == 1:  # Single detection (shape [4])
            boxes = [boxes]  # Wrap in a list
        if isinstance(score_list, (np.float32, np.float64, np.int32, np.int64)):  # Single score
            score_list = [score_list]  # Wrap in a list

        # Draw each detection on the image
        for box, score in zip(boxes, score_list):
            if isinstance(box, (np.ndarray, list)) and len(box) == 4:  # Ensure it's a valid box
                x, y, w, h = map(int, box)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                print(f"Invalid box format: {box}")

        # Display the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(filename)
        plt.axis('off')
        plt.show()

# Run the function
display_images_with_detections()
