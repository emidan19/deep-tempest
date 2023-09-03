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
        self.sigma_test = opt['sigma_test'] if opt['sigma_test']  else 0
        self.use_all_patches = opt['use_all_patches'] if opt['use_all_patches'] else False
        self.num_patches_per_image = opt['num_patches_per_image'] if opt['num_patches_per_image'] else 100

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        # Repeat every image in path list to get more than one patch per image
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
        L_path = H_path

        img_H = util.imread_uint(H_path, self.n_channels_datasetload)    

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H/M patch pairs
            # --------------------------------
            """
            H, W = img_H.shape[:2]

            if self.use_all_patches or (img_H.shape[0] <= self.patch_size) or (img_H.shape[1] <= self.patch_size):

                # ---------------------------------
                # Start or continue image patching
                # ---------------------------------                
                img_patch_index = index % self.num_patches_per_image  # Resets to 0 every time index overflows num_patches
                
                # Upper-left corner of patch
                h_index = self.patch_size * ( (img_patch_index * self.patch_size) // W)
                w_index =  self.patch_size * ( ( (img_patch_index * self.patch_size) % W ) // self.patch_size)

                # Dont exceed the image limit
                h_index = min(h_index, H - self.patch_size)
                w_index = min(w_index, W - self.patch_size)


            else:
                # ---------------------------------
                # randomly crop the patch
                # ---------------------------------
                h_index = random.randint(0, max(0, H - self.patch_size))
                w_index = random.randint(0, max(0, W - self.patch_size))

            # Ground-truth as channels mean
            patch_H = np.mean(img_H[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size, :],axis=2)
            
            # Get the low-resolution patch
            patch_L = patch_H.copy()

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

            # ---------------------------------
            # get noise level
            # ---------------------------------
            noise_level = torch.FloatTensor([int(np.random.uniform(self.sigma_min, self.sigma_max))])/255.0
            # noise_level = torch.FloatTensor([np.random.randint(self.sigma_min, self.sigma_max)])/255.0
            if (self.sigma_max != 0):
                # ---------------------------------
                # add noise
                # ---------------------------------
                noise = torch.randn(img_L.size()).mul_(noise_level).float()
                img_L.add_(noise)
                img_L = torch.cat((img_L, noise), dim=0)

        else:
            """
            # -------------------train_loader-------------
            # get L/H/sigma image pairs
            # --------------------------------
            """

            # Ground-truth as mean value of RGB channels
            img_H = np.mean(img_H,axis=2)
            img_H = img_H[:,:,np.newaxis]
            img_L = img_H.copy()

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

            #img_H = util.uint2single(img_H)

            #img_L = util.uint2single(img_L)

            
            # ---------------------------------
            # get noise level
            # ---------------------------------

            noise_level = torch.FloatTensor([int(self.sigma_test)])/255.0
            if self.sigma_test != 0:

                # noise_level = torch.FloatTensor([np.random.randint(self.sigma_min, self.sigma_max)])/255.0
            
                # ---------------------------------
                # add noise
                # ---------------------------------
                noise = torch.randn(img_L.size()).mul_(noise_level).float()
                img_L.add_(noise)
                img_L = torch.cat((img_L, noise), dim=0)



        noise_level = noise_level.unsqueeze(1).unsqueeze(1)


        return {'L': img_L, 'H': img_H, 'C': noise_level, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
