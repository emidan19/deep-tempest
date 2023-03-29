import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import itertools


class DatasetFFDNet(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H/M for denosing on AWGN with a range of sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., FFDNet, H = f(L, sigma), sigma is noise level
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetFFDNet, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.n_channels_datasetload = opt['n_channels_datasetload'] if opt['n_channels_datasetload'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else [0, 75]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 25
        self.num_patches_per_image = opt['num_patches_per_image']

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        if self.opt['phase'] == 'train':
            listOfLists = [list(itertools.repeat(path, self.num_patches_per_image)) for path in self.paths_H]
            self.paths_H = list(itertools.chain.from_iterable(listOfLists))

            listOfLists = [list(itertools.repeat(path, self.num_patches_per_image)) for path in self.paths_L]
            self.paths_L = list(itertools.chain.from_iterable(listOfLists))

    def __getitem__(self, index):

        # -------------------------------------
        # get H and L image
        # -------------------------------------
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]

        H_file, L_file = H_path.split('/')[-1], L_path.split('/')[-1]
        H_name, L_name = H_file.split('.')[0], L_file.split('.')[0]
        
        assert H_name==L_name, 'Both high and low quality images MUST have same name'

        img_H = util.imread_uint(H_path, self.n_channels_datasetload)       

        # L_path = H_path
        
        img_L = util.imread_uint(L_path, self.n_channels_datasetload)[:,:,:2]       

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H/M patch pairs
            # --------------------------------
            """
            H, W = img_H.shape[:2]

            # ---------------------------------
            # randomly crop the patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))

            # # Comment because using one only channel to train as ground-truth
            # patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # Ground-truth as channels mean
            patch_H = np.mean(img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :],axis=2)
            
            # Get the patch from the simulation
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # # Commented augmentation with rotating because of TMDS encoding

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            # mode = random.randint(0, 7)
            # patch_L = util.augment_img(patch_H, mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

            # ---------------------------------
            # get noise level
            # ---------------------------------
            # noise_level = torch.FloatTensor([np.random.randint(self.sigma_min, self.sigma_max)])/255.0
            noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0

            # ---------------------------------
            # add noise
            # ---------------------------------
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)

        else:
            """
            # --------------------------------
            # get L/H/sigma image pairs
            # --------------------------------
            """

            # Ground-truth as mean value of RGB channels
            img_H = np.mean(img_H,axis=2)
            img_H = img_H[:,:,np.newaxis]
            img_H = util.uint2single(img_H)

            np.random.seed(seed=0)
            img_L = img_L + np.random.normal(0, self.sigma_test/255.0, img_L.shape)
            noise_level = torch.FloatTensor([self.sigma_test/255.0])

            # ---------------------------------
            # L/H image pairs
            # ---------------------------------
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        noise_level = noise_level.unsqueeze(1).unsqueeze(1)


        return {'L': img_L, 'H': img_H, 'C': noise_level, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
