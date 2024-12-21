import cv2 as cv
import numpy as np

def create_gaussian_pyramid_with_scales(img, scale_factor=0.5, min_size=(64, 64)):
    """
    Creează o piramidă gaussiană și păstrează factorii de scalare pentru fiecare nivel.
    :param img: Imaginea originală (numpy array).
    :param scale_factor: Factorul de reducere a dimensiunii la fiecare nivel.
    :param min_size: Dimensiunea minimă la care să se oprească redimensionarea.
    :return: O listă de imagini redimensionate și lista de factori de scalare.
    """
    pyramid = [img]
    scales = [1.0]  # Primul nivel are factorul de scalare 1 (imaginea originală)
    while img.shape[0] > min_size[0] and img.shape[1] > min_size[1]:
        img = cv.pyrDown(img)
        pyramid.append(img)
        scales.append(scales[-1] * scale_factor)  # Actualizează factorul de scalare
    return pyramid, scales

def adjust_coordinates_to_original(coords, scale):
    """
    Ajustează coordonatele de la o imagine redimensionată la dimensiunea originală.
    :param coords: Lista coordonatelor detectate (x_min, y_min, x_max, y_max).
    :param scale: Factorul de scalare utilizat la nivelul curent.
    :return: Lista coordonatelor ajustate.
    """
    adjusted_coords = []
    for coord in coords:
        x_min, y_min, x_max, y_max = coord
        adjusted_coords.append([
            int(x_min / scale),
            int(y_min / scale),
            int(x_max / scale),
            int(y_max / scale)
        ])
    return adjusted_coords

# Exemplu de utilizare
img = cv.imread(r'C:\Users\User\Desktop\university\CAVA-TEMA-2\validare\validare_20\003.jpg', cv.IMREAD_GRAYSCALE)
# Creează piramida gaussiană și păstrează scalele
pyramid, scales = create_gaussian_pyramid_with_scales(img)

all_detections = []  # Pentru detectiile finale
for i, resized_img in enumerate(pyramid):
    scale = scales[i]
    
    # Aici faceți procesarea HOG și detectările
    # Ex. detectii = [[50, 60, 100, 120], [30, 40, 70, 80]]
    detectii = [[50, 60, 100, 120], [30, 40, 70, 80]]  # Exemplu coordonate
    
    # Ajustați coordonatele la dimensiunea originală
    adjusted = adjust_coordinates_to_original(detectii, scale)
    all_detections.extend(adjusted)

# Acum toate coordonatele sunt în raport cu imaginea originală
print(all_detections)