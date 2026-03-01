from scipy.io import loadmat

def get_Farmland_dataset():
    data_set_before = loadmat(r'./datasets/HZB_dataset/image1_ZY.mat')['image1_ZY']
    data_set_after = loadmat(r'./datasets/HZB_dataset/image2_ZY.mat')['image2_ZY']
    ground_truth = loadmat(r'./datasets/HZB_dataset/GT_01_400')['GT_01_400']

    img1 = data_set_before.astype('float32')
    img2 = data_set_after.astype('float32')
    gt = ground_truth.astype('float32')

    return img1, img2, gt


def get_dataset(current_dataset):
    if current_dataset == 'HZB':
        return get_Farmland_dataset()

