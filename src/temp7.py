import cv2 as cv
import os

def create_gaussian_pyramid_with_scales(img, scale_factor=0.8, min_size=(64, 64)):
    """
    Creează o piramidă Gaussiană folosind un factor de scalare personalizat.

    :param img: Imaginea inițială (numpy array).
    :param scale_factor: Factorul de scalare pentru fiecare nivel din piramidă.
    :param min_size: Dimensiunea minimă permisă pentru imagini (în pixeli).
    :return: (pyramid, scales) - imagini la diverse scale și scale-urile asociate.
    """
    pyramid = [img]
    scales = [1.0]

    while img.shape[0] > min_size[0] and img.shape[1] > min_size[1] and scales[-1] * scale_factor >= 0.25:
        # Calculează noua dimensiune pe baza scale factor-ului
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)

        # Redimensionează imaginea cu cv.resize
        img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)

        # Adaugă imaginea redimensionată la piramidă
        pyramid.append(img)

        # Actualizează scale-urile
        scales.append(scales[-1] * scale_factor)

    return pyramid, scales

if __name__ == "__main__":
    # Calea imaginii
    image_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare\mom\0002.jpg"

    # Verifică dacă imaginea există
    if not os.path.exists(image_path):
        print(f"Eroare: Imaginea nu a fost găsită la calea {image_path}")
        exit(1)

    # Citește imaginea
    img = cv.imread(image_path)

    # Verifică dacă imaginea a fost citită corect
    if img is None:
        print("Eroare: Imaginea nu a putut fi citită.")
        exit(1)

    # Creează piramida Gaussiană
    scale_factor = 0.8
    min_size = (64, 64)
    pyramid, scales = create_gaussian_pyramid_with_scales(img, scale_factor, min_size)

    # Creează o cale de ieșire pentru salvarea rezultatelor
    output_dir = os.path.join('./output', "pyramid_results")
    os.makedirs(output_dir, exist_ok=True)

    # Salvează imaginile generate și scale-urile
    for i, (scaled_img, scale) in enumerate(zip(pyramid, scales)):
        output_path = os.path.join(output_dir, f"level_{i}_scale_{scale:.2f}.jpg")
        cv.imwrite(output_path, scaled_img)

    print(f"Piramida Gaussiană generată și salvată în {output_dir}")
