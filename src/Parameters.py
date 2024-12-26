import os

class Parameters:
    def __init__(self):
        self.base_dir = './output'
        self.input_dir = './antrenare'


        self.task1_solution_dir = os.path.join(self.base_dir, 'solution', 'task1')
        self.task2_solution_dir = os.path.join(self.base_dir, 'solution', 'task2')

        self.dir_pos_examples = os.path.join(self.base_dir, 'pozitive')
        self.dir_neg_examples = os.path.join(self.base_dir, 'negative')
        self.dir_hard_neg_examples = os.path.join(self.base_dir, 'hard-negative')

        self.dir_train_examples = r'C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare'
        self.train_adnotations = r'C:\Users\User\Desktop\university\CAVA-TEMA-2\antrenare'

        self.dir_test_examples = r'C:\Users\User\Desktop\university\CAVA-TEMA-2\validare\validare_20'
        # self.path_annotations = r'C:\Users\User\Desktop\university\CAVA-TEMA-2\validare\custom_annotations.txt'
        self.path_annotations = r'C:\Users\User\Desktop\university\CAVA-TEMA-2\validare\task1_gt_validare20.txt'
        self.dir_save_files = os.path.join(self.base_dir, 'metrics')

        self.aspect_ratios = [0.8, 1.0, 1.2, 1.4]
        self.characters = ['dad', 'mom', 'dexter', 'deedee']

        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 81  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 8  # dimensiunea celulei
        self.overlap = 0.3
        self.number_positive_examples = 5665  # numarul exemplelor pozitive
        self.number_negative_examples = 17660  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = True
        self.threshold = 0
        self.bins = 6
        self.scale = 0.8
