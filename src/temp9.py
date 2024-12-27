import numpy as np

# Path to the .npy files
detections_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\evaluare\fisiere_solutie\331_Alexe_Bogdan\task1\detections_all_faces.npy"
scores_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\evaluare\fisiere_solutie\331_Alexe_Bogdan\task1\scores_all_faces.npy"

try:
    # Load the .npy files
    detections = np.load(detections_path)
    scores = np.load(scores_path)

    # Check if the lengths match
    if len(detections) != len(scores):
        raise ValueError("Detections and scores arrays must have the same length.")

    # Combine detections and scores into a single array
    combined = np.hstack((detections, scores.reshape(-1, 1)))

    # Find unique rows based on the bounding box and score
    unique_combined = np.unique(combined, axis=0)

    # Sort unique rows by the score column (last column)
    sorted_combined = unique_combined[np.argsort(-unique_combined[:, -1])]

    # Display sorted unique detections with width, height, and scores
    print("Sorted unique detections with width, height, and scores:")
    for row in sorted_combined:
        width = row[2] - row[0]
        height = row[3] - row[1]
        print(f"Bounding Box: {row[:4].tolist()}, Width: {width}, Height: {height}, Score: {row[-1]:.4f}")

except Exception as e:
    print("An error occurred:", e)
