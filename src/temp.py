import os
from collections import defaultdict
import numpy as np

# Define file paths
file_paths = [
    r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\dad_annotations.txt",
    r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\deedee_annotations.txt",
    r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\dexter_annotations.txt",
    r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\mom_annotations.txt"
]

# Function to read and compute aspect ratios from files
def count_aspect_ratios(file_paths):
    aspect_ratios = []
    
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
                            aspect_ratios.append(aspect_ratio)
        else:
            print(f"File not found: {file_path}")
    
    return aspect_ratios

# Function to bin aspect ratios
def bin_aspect_ratios(aspect_ratios, num_bins):
    min_ratio = min(aspect_ratios)
    max_ratio = max(aspect_ratios)
    bins = np.linspace(min_ratio, max_ratio, num_bins + 1)
    bin_counts, bin_edges = np.histogram(aspect_ratios, bins=bins)
    return bin_counts, bin_edges

# Calculate aspect ratios
aspect_ratios = count_aspect_ratios(file_paths)

# Bin aspect ratios
num_bins = 6  # Define the number of bins
bin_counts, bin_edges = bin_aspect_ratios(aspect_ratios, num_bins)

# Display the bins and their counts
for i in range(len(bin_counts)):
    print(f"Bin {i + 1} ({bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}): {bin_counts[i]} values")
