import os

def count_images_in_folder(folder_path):
    # Definim extensiile comune pentru fișierele imagine
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    
    try:
        # Obținem o listă cu toate fișierele din folder
        files = os.listdir(folder_path)
        
        # Filtrăm doar fișierele cu extensii de imagine
        image_files = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
        
        # Returnăm numărul de imagini
        return len(image_files)
    except Exception as e:
        print(f"A apărut o eroare: {e}")
        return 0

# Specificăm calea către folderul dorit
folder_path = r"C:\Users\User\Desktop\university\CAVA-TEMA-2\output\hard-negative\ratio_10"

# Apelăm funcția și afișăm rezultatul
num_images = count_images_in_folder(folder_path)
print(f"Numărul de imagini din folder este: {num_images}")
