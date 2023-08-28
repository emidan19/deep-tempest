import os
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from utils.DTutils import is_natural_patch
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
        self.skip_natural_patches = opt['skip_natural_patches'] if opt['skip_natural_patches'] else False

        # -------------------------------------
        # Dataset path contains all H images and subfolders for every single one with one or more
        # -------------------------------------

        """  
        Dataset path includes all H images and L-subfolders.
        Every H image has one L-subfolder assosiated, which
        contains one or more L representations of the H image.
        """


        self.paths_H = [f for f in os.listdir(opt['dataroot_H']) if os.path.isfile(f)]
        #------------------------------------------------------------------------------------------------------
        # For the above step one can use util.get_image_paths(), but it goes recursevely thought the tree dirs
        #------------------------------------------------------------------------------------------------------

        paths_H_aux = []
        self.paths_L = []

        # Iterate over all image paths
        for H_file in self.paths_H:
            # filename = os.path.basename(H_file)
            filename = H_file.split(".png") # TODO: the correct way to do it is with os.path.basename()
            L_folder = os.path.join(opt['dataroot_H'],filename)
            # For image at subfolder, append to L paths and repeat current H path
            for L_file in os.listdir(L_folder):
                L_filepath = os.path.join(L_folder,L_file)
                paths_H_aux.append(H_file)
                self.paths_L.append(L_filepath)

        # Update H paths
        self.paths_H = paths_H_aux
        
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
        L_path = self.paths_L[index]

        H_file, L_file = H_path.split('/')[-1], L_path.split('/')[-1]
        H_name, L_name = H_file.split('.')[0], L_file.split('.')[0]
        
        assert H_name==L_name, f'Both high and low quality images MUST have same name.\nGot {H_name} and {L_name} respectively.'

        img_H = util.imread_uint(H_path, self.n_channels_datasetload)       

        
        img_L = util.imread_uint(L_path, self.n_channels_datasetload)[:,:,:2]       

        # Temp solution for blanking images
        L_v, L_h = img_L.shape[:2]

        if L_v==1000 and L_h==1800:
          img_L  = img_L[(1000-900)//2:-(1000-900)//2,(1800-1600)//2:-(1800-1600)//2,:]

        # Get module of complex image, stretch and to uint8
        # img_L = img_L.astype('float')
        # img_L = np.abs(img_L[:,:,0]+1j*img_L[:,:,1])
        # img_L = 255*(img_L - img_L.min())/(img_L.max() - img_L.min())
        # img_L = img_L.astype('uint8')

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

                ### Keep text patches only (non-natural images)
                if self.skip_natural_patches:

                    # Check if selected patch is natural, based on RGB entropy
                    is_natural = is_natural_patch(img_H[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size, :])

                    # If natural, select random patch and keep trying until non-natural or reaching max attempts
                    attempt = 0
                    max_attempts = 10
                    while is_natural and (attempt < max_attempts):
                        h_index = random.randint(0, max(0, H - self.patch_size))
                        w_index = random.randint(0, max(0, W - self.patch_size))
                        is_natural = is_natural_patch(img_H[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size, :])
                        attempt += 1


            else:
                # ---------------------------------
                # randomly crop the patch
                # ---------------------------------
                h_index = random.randint(0, max(0, H - self.patch_size))
                w_index = random.randint(0, max(0, W - self.patch_size))

            # Ground-truth as channels mean
            patch_H = np.mean(img_H[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size, :],axis=2)
            
            # Get the patch from the simulation
            patch_L = img_L[h_index:h_index + self.patch_size, w_index:w_index + self.patch_size,:]

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

        else:
            """
            # --------------------------------
            # get L/H/sigma image pairs
            # --------------------------------
            """

            # Ground-truth as mean value of RGB channels
            img_H = np.mean(img_H,axis=2)
            img_H = img_H[:,:,np.newaxis]

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

            # img_H = util.uint2single(img_H)

            # img_L = util.uint2single(img_L)

            
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



        noise_level = noise_level.unsqueeze(1).unsqueeze(1)


        return {'L': img_L, 'H': img_H, 'C': noise_level, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
