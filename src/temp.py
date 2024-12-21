import os
from collections import defaultdict

# Define file paths
file_paths = [
    r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\dad_annotations.txt",
    r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\deedee_annotations.txt",
    r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\dexter_annotations.txt",
    r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\mom_annotations.txt"
]

# Function to read and compute aspect ratios from files
def count_aspect_ratios(file_paths):
    aspect_ratios = defaultdict(int)
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x1, y1, x2, y2 = map(int, parts[1:5])
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)
                        if width > 0 and height > 0:
                            aspect_ratio = round(width / height, 2)
                            aspect_ratios[aspect_ratio] += 1
        else:
            print(f"File not found: {file_path}")
    
    return dict(aspect_ratios)

# Calculate and display aspect ratios
aspect_ratios = count_aspect_ratios(file_paths)
print(aspect_ratios)
