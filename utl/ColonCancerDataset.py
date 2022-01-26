import os
import numpy as np
import scipy.io
import glob
from PIL import Image
import multiprocessing
from utl.data_aug_op import hematoxylin_eosin_aug, normalize

class ColonCancerDataset(object):

    def __init__(self, patch_size,seed=None,augmentation=True, **kwargs):

        self.patch_size = patch_size
        self.augmentation=augmentation
        self.seed=seed

        super(ColonCancerDataset, self).__init__(**kwargs)


    def preprocess_bags(self,wsi_paths):
        wsi = []

        for each_path in wsi_paths:

            mat_files = glob.glob(os.path.join(os.path.split(each_path)[0], "*epithelial.mat"))

            epithelial_file = scipy.io.loadmat(mat_files[0])
            label = 0 if (epithelial_file["detection"].size == 0) else 1

            img_data = np.asarray(Image.open(each_path), dtype=np.float32)
            wsi.append((img_data.astype(np.float32), label, each_path))

        return wsi

    def load_bags(self,wsi_paths):
        wsi = self.preprocess_bags(wsi_paths)

        bags = []
        for ibag, bag in enumerate(wsi):

            num_ins = 0
            img = []
            name_img = []

            for enum, cell_type in enumerate(['epithelial', 'fibroblast', 'inflammatory', 'others']):

                dir_cell = os.path.splitext(bag[2])[0] + '_' + cell_type + '.mat'

                with open(dir_cell, 'rb') as f:
                    mat_cell = scipy.io.loadmat(f)

                    num_ins += len(mat_cell['detection'])

                    for (x, y) in mat_cell['detection']:

                        x = np.round(x)
                        y = np.round(y)

                        if x < np.floor(self.patch_size / 2):
                            x_start = 0
                            x_end = self.patch_size
                        elif x > 500 - np.ceil(self.patch_size / 2):
                            x_start = 500 - self.patch_size
                            x_end = 500
                        else:
                            x_start = x - np.floor(self.patch_size / 2)
                            x_end = x + np.ceil(self.patch_size / 2)
                        if y < np.floor(self.patch_size / 2):
                            y_start = 0
                            y_end = self.patch_size
                        elif y > 500 - np.ceil(self.patch_size / 2):
                            y_start = 500 - self.patch_size
                            y_end = 500
                        else:
                            y_start = y - np.floor(self.patch_size / 2)
                            y_end = y + np.ceil(self.patch_size / 2)

                        patch = bag[0][int(y_start):int(y_end), int(x_start):int(x_end)]

                        patch = normalize(patch)
                        patch = np.asarray(patch, dtype=np.float32)
                        patch /= 255

                        img.append(np.expand_dims(patch, 0))

                        name_img.append("{}-xpos{}-ypos{}-{}-{}.png".format(
                            os.path.basename(bag[2])[:-4], int(x), int(y), bag[1], cell_type))



            if bag[1] == 1:
                curr_label =1
            else:
                curr_label =0

            stack_img = np.concatenate(img, axis=0)

            assert num_ins == stack_img.shape[0]

            bags.append((stack_img, curr_label, name_img))

        return bags


    def parallel_load_bags(self, wsi_paths):
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            ncpus = multiprocessing.cpu_count()


        with multiprocessing.get_context('spawn').Pool(ncpus)  as pool:

            wsi_paths=[wsi_paths[i:: (ncpus)] for i in range(ncpus)]

            data_set = pool.map(self.load_bags, wsi_paths)
        data_set=np.concatenate(data_set,axis=0)

        return data_set




