import os
import numpy as np

def get_all_ratios(input_folder, characters):
    """
    Parcurge toate fișierele de anotări pentru personajele date
    și returnează o listă cu toate valorile w/h.
    """
    all_ratios = []

    for character in characters:
        # Fișierul de anotări e așteptat să fie `nume_character_annotations.txt`
        annotation_file = os.path.join(input_folder, f'{character}_annotations.txt')

        if not os.path.exists(annotation_file):
            print(f"[WARN] Fișierul de anotări nu există: {annotation_file}")
            continue

        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                if len(line) < 5:
                    continue

                # line[0] = nume imagine, următoarele 4 sunt bounding box
                x_min, y_min, x_max, y_max = map(int, line[1:5])
                w = x_max - x_min
                h = y_max - y_min
                if w > 0 and h > 0:
                    ratio = w / float(h)
                    all_ratios.append(ratio)
    
    return all_ratios

def find_optimal_bins(ratios, n_bins=4):
    """
    Returnează (n_bins - 1) praguri care împart lista de ratio-uri
    în n_bins regiuni cu număr aproximativ egal de exemple.
    """
    ratios_np = np.array(ratios)
    ratios_np.sort()

    # De exemplu, pentru 4 bin-uri calculezi pragurile la 25%, 50%, 75%
    # (quantile la 0.25, 0.5, 0.75).
    # Ultimul bin se încheie la valoarea maximă.
    quantiles = [(i / n_bins) for i in range(1, n_bins)]
    # Asta va genera [0.25, 0.5, 0.75] pentru n_bins=4
    cut_points = np.quantile(ratios_np, quantiles)

    return cut_points

def main():
    # Exemplu de config
    input_folder = 'C:/Users/User/Desktop/university/CAVA-TEMA-2/antrenare'
    characters = ["dad", 'mom', 'dexter', 'deedee']
    n_bins = 3

    # Citește toate ratio-urile
    all_ratios = get_all_ratios(input_folder, characters)
    print(f"Am găsit {len(all_ratios)} bounding box-uri în total.")

    if not all_ratios:
        print("Nu ai deloc bounding box-uri valide, verifică datele.")
        return

    # Află pragurile pentru bin-uri
    cut_points = find_optimal_bins(all_ratios, n_bins=n_bins)
    print(f"Praguri (cut points) pt {n_bins} bin-uri: {cut_points}")

    # Ca exemplu, poți să faci un mic raport de câte intră în fiecare bin:
    bin_counts = [0]*n_bins
    for ratio in all_ratios:
        # Verificăm în ce bin se încadrează ratio-ul
        found_bin = False
        for i in range(n_bins-1):
            if ratio < cut_points[i]:
                bin_counts[i] += 1
                found_bin = True
                break
        if not found_bin:
            bin_counts[-1] += 1

    for i in range(n_bins):
        if i == 0:
            print(f"Bin {i+1}: ratio < {cut_points[i]:.3f} -> {bin_counts[i]} exemple")
        elif i == n_bins - 1:
            print(f"Bin {i+1}: ratio >= {cut_points[i-1]:.3f} -> {bin_counts[i]} exemple")
        else:
            print(f"Bin {i+1}: {cut_points[i-1]:.3f} <= ratio < {cut_points[i]:.3f} -> {bin_counts[i]} exemple")

if __name__ == "__main__":
    main()
