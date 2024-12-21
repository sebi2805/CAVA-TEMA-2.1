import os
import cv2
from collections import Counter

# Define the folder paths
folders = ['output/pozitive/dad']
# , 'output/deedee', 'output/dexter', 'output/mom', 'output/unknown'
# Function to normalize and truncate aspect ratios
def normalize_aspect_ratio(width, height):
    normalized_width = width / width
    normalized_height = height / width

    # Normalize so that at least one value is 1
    if normalized_width > normalized_height:
        normalized_height = 1
        normalized_width = normalized_width / normalized_height
    else:
        normalized_width = 1
        normalized_height = normalized_height / normalized_width

    # Truncate to 2 decimal places
    normalized_width = round(normalized_width, 2)
    normalized_height = round(normalized_height, 2)

    return f"{normalized_width}:{normalized_height}"

# Collect all aspect ratios
aspect_ratios = []

# Loop through each folder and read images
for folder in folders:
    for filename in os.listdir(folder):
        if not filename.startswith("dad"):
            continue
        file_path = os.path.join(folder, filename)
        print(f"Reading {file_path}")
        try:
            # Read the image using OpenCV
            img = cv2.imread(file_path)
            if img is not None:
                height, width, _ = img.shape
                aspect_ratios.append(normalize_aspect_ratio(width, height))
            else:
                print(f"Unable to read {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Count the unique aspect ratios
aspect_ratio_counts = Counter(aspect_ratios)

# Sort aspect ratios by count (highest to lowest)
sorted_aspect_ratios = sorted(aspect_ratio_counts.items(), key=lambda x: x[1], reverse=True)

# Print the results
print("Normalized Aspect Ratios and Their Counts:")
for ratio, count in sorted_aspect_ratios:
    print(f"{ratio}: {count}")
