import os

class Parameters:
    def __init__(self):
        self.base_dir = './output'
        self.dir_pos_examples = os.path.join(self.base_dir, 'pozitive')
        self.dir_neg_examples = os.path.join(self.base_dir, 'negative')
        self.dir_test_examples = r'C:\Users\User\Desktop\university\CAVA-TEMA-2\validare\validare'  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.path_annotations = r'C:\Users\User\Desktop\university\CAVA-TEMA-2\validare\task1_gt_validare.txt'
        self.dir_save_files = os.path.join(self.base_dir, 'metrics')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 64  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 5665  # numarul exemplelor pozitive
        self.number_negative_examples = 17660  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
        self.bins = 4
