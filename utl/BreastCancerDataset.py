import os
from cell_segmenation_util import *


class BreastCancerDataset(object):

    def __init__(self, patch_size,stride,model,format,seed=None,augmentation=True, **kwargs):

        self.stride=stride
        self.patch_size = patch_size
        self.augmentation=augmentation
        self.seed=seed
        self.model=model
        self.format=format
        super(BreastCancerDataset, self).__init__(**kwargs)


    def load_bags(self,wsi_paths):


        bags = []

        for index, temp_name in enumerate(wsi_paths):
            img = []
            name_img = []

            patch_index = 0
            print('process: ', str(index), ' name: ', temp_name)

            temp_image = cv2.imread(temp_name)

            if temp_image is None:
                raise AssertionError(temp_name, ' not found')
            batch_group, shape = preprocess(temp_image, self.patch_size, self.stride, temp_name)
            mask_list = sess_interference(self.model, batch_group)
            c_mask = patch2image(mask_list, self.patch_size, self.stride, shape)
            c_mask = cv2.medianBlur((255 * c_mask).astype(np.uint8), 3)
            c_mask = c_mask.astype(np.float) / 255

            thr = 0.5
            c_mask[c_mask < thr] = 0
            c_mask[c_mask >= thr] = 1
            center_edge_mask, gray_map, center_coordinates = center_edge(c_mask, temp_image)

            patch_prefix = 0 if "benign" in temp_name else 1

            class_name = os.path.basename(temp_name).split(".")[0]

            for enum, center_coordinate in enumerate(center_coordinates):
                try:

                    patch = temp_image[center_coordinate[0] - self.stride:center_coordinate[0] + self.stride,
                            center_coordinate[1] - self.stride:center_coordinate[1] + self.stride]

                    if patch.shape == (32, 32, 3):
                        mask = cv2.inRange(patch, (230, 230, 230), (255, 255, 255))

                        total = patch.shape[0] * patch.shape[1]
                        white = cv2.countNonZero(mask)
                        ratio = white / total
                        if ratio<0.7:
                            patch = np.asarray(patch, dtype=np.float32)
                            patch[:, :, 0] -= 202.812
                            patch[:, :, 1] -= 134.81
                            patch[:, :, 2] -= 164.48

                            patch /= 255


                            img.append(np.expand_dims(patch, 0))
                            name_img.append("{}-xpos{}-ypos{}-{}.png".format(
                                    class_name, int(center_coordinate[0]), int(center_coordinate[1]), patch_prefix
                                ))
                            patch_index += 1
                except:
                    pass
            if patch_prefix == 1:
                curr_label = 1
            else:
                curr_label = 0

            stack_img = np.concatenate(img, axis=0)

            assert patch_index == stack_img.shape[0]

            bags.append((stack_img, curr_label, name_img))

        return bags



