
import os.path
import numpy as np
import random
from data.base_dataset import BaseDataset, get_transform, detect_colour_channel, load_points, inside
from data.image_folder import make_dataset
from PIL import Image
from skimage.color import rgb2gray, rgb2hed
from matplotlib.colors import LinearSegmentedColormap


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.A_dir)
        self.dir_B = os.path.join(opt.dataroot, opt.B_dir)
        if self.opt.register_images:
            self.dir_A_registered_from_B = os.path.join(opt.dataroot, opt.A_registered_from_B_dir)
            self.dir_B_registered_from_A = os.path.join(opt.dataroot, opt.B_registered_from_A_dir)

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        if self.opt.register_images:
            self.A_registered_from_B_paths = make_dataset(self.dir_A_registered_from_B)
            self.B_registered_from_A_paths = make_dataset(self.dir_B_registered_from_A)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        if self.opt.register_images:
            self.A_registered_from_B_paths = sorted(self.A_registered_from_B_paths)
            self.B_registered_from_A_paths = sorted(self.B_registered_from_A_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
        self.shuffled_index = list(range(len(self.A_paths)))
        self.shuffled_index_B = list(range(len(self.B_paths)))
        if opt.shuffle_data:
            random.shuffle(self.shuffled_index)
            random.shuffle(self.shuffled_index_B)

        self.cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white', 'saddlebrown'])

        # Load points from David
        true_label_path = os.path.join(opt.dataroot, opt.A_dir.split("/")[0], opt.true_label_xml)
        self.mitosis_points = load_points(true_label_path)

    def __getitem__(self, index):
        if self.opt.shuffle_data:
            index = self.shuffled_index[index % self.A_size]
        A_path = self.A_paths[index]
        if self.opt.register_images:
            B_registered_from_A_path = self.B_registered_from_A_paths[index]
        if self.opt.shuffle_data:
            index_B = self.shuffled_index_B[index % self.B_size]
        # if self.opt.serial_batches:
        #     index_B = index % self.B_size
        # else:
        #     index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        if self.opt.register_images:
            A_registered_from_B_path = self.A_registered_from_B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        if self.opt.register_images:
            A_registered_from_B_img = Image.open(A_registered_from_B_path).convert('RGB')
            B_registered_from_A_img = Image.open(B_registered_from_A_path).convert('RGB')

        if self.opt.color_conversion == 'dab':
            B_img = self.convert_dab(B_img)
            if self.opt.register_images:
                B_registered_from_A_img = self.convert_dab(B_registered_from_A_img)
        A = self.transform(A_img)
        B = self.transform(B_img)
        if self.opt.register_images:
            A_registered_from_B = self.transform(A_registered_from_B_img)
            B_registered_from_A = self.transform(B_registered_from_A_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        # Calculate label
        AtoB = self.opt.which_direction == 'AtoB'
        # real_dab_img = self.convert_dab_only(A_img if AtoB else B_registered_from_A_img)
        if AtoB or (not AtoB and self.opt.phase == 'test'):
            # if AtoB take reference label from real B image (=PHH3) registered to A image, and name from real A image (=H&E).
            detect = detect_colour_channel(B_registered_from_A_img,
                                           channel=2, min=-0.9, max=None, threshold=0.5, count=400, conversion='rgb2hed')
            real_name = os.path.split(A_path)[1].replace('.png', '')
        else:
            # if BtoA take reference label from real B image (=PHH3), and name from real B image.
            detect = detect_colour_channel(B_img,
                                           channel=2, min=-0.9, max=None, threshold=0.5, count=400, conversion='rgb2hed')
            real_name = os.path.split(B_path)[1].replace('.png', '')

        real_label = np.zeros((2, 1, 1)).astype(np.float32)
        real_label[detect] = 1.
        mitosis_points = inside(real_name, self.mitosis_points)
        true_label = np.zeros((2, 1, 1)).astype(np.float32)
        true_label[int(mitosis_points.size > 2)] = 1.

        if self.opt.register_images:
            dataset = {'A': A, 'B': B,
                       'A_registered_from_B': A_registered_from_B, 'B_registered_from_A': B_registered_from_A,
                       'A_paths': A_path, 'B_paths': B_path,
                       'A_registered_from_B_paths': A_registered_from_B_path,
                       'B_registered_from_A_paths': B_registered_from_A_path,
                       'real_name': real_name, 'real_label': real_label,
                       'mitosis_points': mitosis_points, 'true_label': true_label}
        else:
            dataset = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,
                       'real_name': real_name, 'real_label': real_label,
                       'mitosis_points': mitosis_points, 'true_label': true_label}

        return dataset

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
