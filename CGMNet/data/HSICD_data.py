import numpy as np
import torch
import torch.utils.data as data
import scipy.io as sio

class HSICD_data(data.Dataset):
    def __init__(self, data_sample, cfg):

        self.phase = cfg['phase']
        self.img1 = data_sample['img1_pad']
        self.img2 = data_sample['img2_pad']
        self.patch_coordinates = data_sample['patch_coordinates']
        self.gt = data_sample['img_gt']

        if self.phase == 'train':
            self.data_indices = data_sample['train_sample_center']
        elif self.phase == 'test':
            self.data_indices = data_sample['test_sample_center']

    def create_sample_mask(self, data_sample, indices, is_train=True):
        sample_mask = torch.zeros_like(data_sample['img_gt'], dtype=torch.uint8)
        for index in indices:
            img_index = self.patch_coordinates[index[0]]
            sample_mask[img_index[0]:img_index[1], img_index[2]:img_index[3]] = 1

        if is_train:
            train_sample_mask = sample_mask
            train_sample_mask = np.array(train_sample_mask)
            sio.savemat('train_sample_mask.mat', {'train_sample_mask': train_sample_mask})
        else:
            test_sample_mask = sample_mask
            test_sample_mask = np.array(test_sample_mask)
            sio.savemat('test_sample_mask.mat', {'test_sample_mask': test_sample_mask})

        return sample_mask

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        index = self.data_indices[idx]
        img_index = self.patch_coordinates[index[0]]
        img1 = self.img1[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        img2 = self.img2[:, img_index[0]:img_index[1], img_index[2]:img_index[3]]
        label = self.gt[index[1], index[2]]

        return img1, img2, label, index
