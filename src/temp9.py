import numpy as np

# Path to the .npy file
detections_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\evaluare\fisiere_solutie\331_Alexe_Bogdan\task1\detections_all_faces.npy"

try:
    # Load the .npy file
    detections = np.load(detections_path)

    # Get unique bounding boxes
    unique_bounding_boxes = set(map(tuple, detections))

    # Compute unique width-height pairs and aspect ratios
    unique_dimensions = set()
    unique_aspect_ratios = set()

    for box in unique_bounding_boxes:
        width = box[2] - box[0]
        height = box[3] - box[1]
        aspect_ratio = width / height if height != 0 else float('inf')
        unique_dimensions.add((width, height))
        unique_aspect_ratios.add(round(aspect_ratio, 2))

    # Sort the unique dimensions and aspect ratios
    sorted_dimensions = sorted(unique_dimensions, key=lambda x: (x[0], x[1]))
    sorted_aspect_ratios = sorted(unique_aspect_ratios)

    # Display the results
    print("Shape of the detections array:", detections.shape)
    print("Unique bounding boxes, dimensions, and aspect ratios:")
    # for box in unique_bounding_boxes:
    #     width = box[2] - box[0]
    #     height = box[3] - box[1]
    #     aspect_ratio = width / height if height != 0 else float('inf')
    #     print(f"(np.int32({box[0]}), np.int32({box[1]}), np.int32({box[2]}), np.int32({box[3]})) - Width: {width}, Height: {height}, Aspect Ratio: {aspect_ratio:.2f}")

    # Display sorted unique width-height pairs
    print("\nUnique width-height pairs:")
    for dim in sorted_dimensions:
        print(f"Width: {dim[0]}, Height: {dim[1]}")

    # Display sorted unique aspect ratios
    print("\nUnique aspect ratios:")
    for ratio in sorted_aspect_ratios:
        print(f"Aspect Ratio: {ratio}")

except Exception as e:
    print("An error occurred:", e)
