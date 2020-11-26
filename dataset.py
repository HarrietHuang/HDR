from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
# from utils import is_image_file, load_img
import os
import numpy as np
import cv2
import argparse
import glob
from PIL import Image
from torchvision import transforms as T
from torchvision import transforms
import IPython.display as display

import torch.utils.data as data
import torch
import numpy as np
from skimage.transform import pyramid_gaussian, resize
# teacher
import numpy as np
from skimage.transform import pyramid_gaussian
import cv2
from scipy import signal
import sys
# from cv2.ximgproc import guidedFilter
import random
from skimage.color import rgb2ycbcr
from skimage.transform import pyramid_gaussian, resize
from skimage.io import imsave
from skimage import img_as_ubyte


def adjust_gamma(image, gamma=1):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def upsample(mono):
    img_shape = mono.shape
    if len(img_shape) == 3 and img_shape[2] != 1:
        sys.exit('failure - upsample')

    C = np.zeros([img_shape[0] * 2, img_shape[1] * 2])
    C[1::2, 1::2] = mono
    t1 = list([[0.1250, 0.5000, 0.7500, 0.5000, 0.1250]])
    t2 = list([[0.1250], [0.5000], [0.7500], [0.5000], [0.1250]])
    myj = signal.convolve2d(C, t1, mode="same")
    myj = signal.convolve2d(myj, t2, mode="same")
    return myj


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def la_filter(mono):
    img_shape = mono.shape
    C = np.zeros(img_shape)
    t1 = list([[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]])
    # for i in range(0, img_shape[0]):
    #     for j in range(0, img_shape[1]):
    #         C[i, j] = abs(np.sum(mono[i:i + 3, j:j + 3] * t1))
    myj = signal.convolve2d(mono, t1, mode="same")
    return myj


def contrast(I, exposure_num, img_rows, img_cols):
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        mono = rgb2gray(I[i])
        C[:, :, i] = la_filter(mono)

    return C


def saturation(I, exposure_num, img_rows, img_cols):
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        R = I[i][:, :, 0]
        G = I[i][:, :, 1]
        B = I[i][:, :, 2]
        mu = (R + G + B) / 3
        C[:, :, i] = np.sqrt(
            ((R - mu) ** 2 + (G - mu) ** 2 + (B - mu) ** 2) / 3)
    return C


def well_exposedness(I, exposure_num, img_rows, img_cols):
    sig = 0.2
    n = exposure_num
    C = np.zeros((img_rows, img_cols, n))
    for i in range(0, n):
        R = np.exp(-.4 * (I[i][:, :, 0] - 0.5) ** 2 / sig ** 2)
        G = np.exp(-.4 * (I[i][:, :, 1] - 0.5) ** 2 / sig ** 2)
        B = np.exp(-.4 * (I[i][:, :, 2] - 0.5) ** 2 / sig ** 2)
        C[:, :, i] = R * G * B
    return C


def gaussian_pyramid(I, nlev, multi):
    pyr = []

    # for ii in range(0,nlev):
    #     temp = pyramid_gaussian(I, downscale=2)
    #     pyr.append(temp)
    for (i, resized) in enumerate(pyramid_gaussian(I, downscale=2, multichannel=multi)):
        if i == nlev:
            break
        pyr.append(resized)
    return pyr


def laplacian_pyramid(I, nlev, mult=True):
    pyr = []
    expand = []
    pyrg = gaussian_pyramid(I, nlev, multi=mult)
    for i in range(0, nlev - 1):

        # expand_temp = cv2.resize(pyrg[i + 1], (pyrg[i].shape[1],
        # pyrg[i].shape[0]))
        expand_temp = resize(
            pyrg[i + 1], (pyrg[i].shape[1], pyrg[i].shape[0]), preserve_range=True, anti_aliasing=False)
        # expand_temp = resize(pyrg[i + 1], (pyrg[i].shape[1], pyrg[i].shape[0]), preserve_range=True, anti_aliasing=False)
        # for j in range(3):
        # expand_temp[:,:,j] = upsample(pyrg[i+1][:,:,j])
        temp = pyrg[i] - expand_temp
        expand.append(expand_temp)
        pyr.append(temp)
    pyr.append(pyrg[nlev - 1])
    expand.append(pyrg[nlev - 1])
    return pyr, expand


def reconstruct_laplacian_pyramid(pyr):
    nlev = len(pyr)
    # print('nlev', nlev)
    R = pyr[nlev - 1]
    for i in range(nlev - 2, -1, -1):
        # R = pyr[i+1]
        odd = R.shape
        # print('odd ',odd)

        # C = np.zeros([odd[0]*2, odd[1]*2, odd[2]])
        # for j in range(odd[2]):
            # print('R', R.shape)
            # print('C', C.shape)
            # print('pyr', pyr[i][:,:,j].shape)
        # upsample(R[:,:,j])#
        C = pyr[i] + cv2.resize(R, (pyr[i].shape[1], pyr[i].shape[0]))
            # C[:,:,j] = pyr[i][:,:,j]  + cv2.resize(R,(pyr[i].shape[1],
            # pyr[i].shape[0]))#upsample(R[:,:,j])#
        R = C
    return R


def Gaussian1D(cen, std, YX1):
    y = np.zeros((1, YX1))
    for i in range(0, YX1):
        y[0][i] = np.exp(-((i - cen)**2) / (2 * (std**2)))
    y = np.round(y * (YX1 - 1))
    return y


def gaussian_pyramid(I, nlev, multi):
    pyr = []

    # for ii in range(0,nlev):
    #     temp = pyramid_gaussian(I, downscale=2)
    #     pyr.append(temp)
    for (i, resized) in enumerate(pyramid_gaussian(I, downscale=2, multichannel=multi)):
        if i == nlev:
            break
        pyr.append(resized)
    return pyr


def laplacian_pyramid(I, nlev, mult=True):
    pyr = []
    expand = []
    pyrg = gaussian_pyramid(I, nlev, multi=mult)
    for i in range(0, nlev - 1):

        # expand_temp = cv2.resize(pyrg[i + 1], (pyrg[i].shape[1],
        # pyrg[i].shape[0]))
        expand_temp = resize(
            pyrg[i + 1], (pyrg[i].shape[1], pyrg[i].shape[0]), preserve_range=True, anti_aliasing=False)
        temp = pyrg[i] - expand_temp
        expand.append(expand_temp)
        pyr.append(temp)
    pyr.append(pyrg[nlev - 1])
    expand.append(pyrg[nlev - 1])
    return pyr, expand


def cfusion(uexp, oexp):
    beta = 2
    vFrTh = 0.16
    RadPr = 3

    I = (uexp, oexp)
    r = uexp.shape[0]
    c = uexp.shape[1]
    n = 2
    nlev = round(np.log(min(r, c)) / np.log(2)) - beta
    nlev = int(nlev)
    RadFr = RadPr * (1 << (nlev - 1))

    W = np.ones((r, c, n))

    W = np.multiply(W, contrast(I, n, r, c))
    W = np.multiply(W, saturation(I, n, r, c))
    W = np.multiply(W, well_exposedness(I, n, r, c))

    W = W + 1e-12
    Norm = np.array([np.sum(W, 2), np.sum(W, 2)])
    Norm = Norm.swapaxes(0, 2)
    Norm = Norm.swapaxes(0, 1)
    W = W / Norm

    II = (uexp / 255.0, oexp / 255.0)

    pyr = gaussian_pyramid(np.zeros((r, c, 3)), nlev, multi=True)
    for i in range(0, n):
        pyrw = gaussian_pyramid(W[:, :, i], nlev, multi=False)
        pyri, content = laplacian_pyramid(II[i], nlev, mult=True)
        for ii in range(0, nlev):
            w = np.array([pyrw[ii], pyrw[ii], pyrw[ii]])
            w = w.swapaxes(0, 2)
            w = w.swapaxes(0, 1)
            pyr[ii] = pyr[ii] + w * pyri[ii]
    R = reconstruct_laplacian_pyramid(pyr)
    # R = cv2.cvtColor(R.astype(np.float32), cv2.COLOR_YCR_CB2BGR)
    # R = ycbcr2rgb(R)

    # R = R * 255
    return R


class HDRdatasets_dynamic_compose(data.Dataset):

    def __init__(self, train=True, transforms=None):
        out_img_train = []
        gt_img_train = []
        img = []
        if train:
            for i in os.listdir(r'C:\Users\admin\Downloads\random_select'):
                img_list = glob.glob(
                    r'C:\Users\admin\Downloads\random_select' + '\\' + i + r'\*.JPG')
                # print(img_list)
#                 img_list1 = glob.glob('.\matlab_compose'+'\\'+i+r'\*.png')
                for j in img_list:
                    # print(j)
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)

#                 for j in img_list1:
#                     if 'com.png' in j:
#                         out_img_train.append(j)
#                     else:
#                         img.append(j)

        else:
            for i in os.listdir(r'C:\Users\admin\Downloads\random_select2'):
                img_list = glob.glob(
                    r'C:\Users\admin\Downloads\random_select2' + '\\' + i + r'\*.JPG')
                img_list1 = glob.glob(
                    r'C:\Users\admin\Downloads\random_select2' + '\\' + i + r'\*.png')
                for j in img_list:
                    # print(j)
                    if 'GT.JPG' in j:
                        gt_img_train.append(j)
                    else:
                        img.append(j)
#                 gt_img_train.append([])
#                 img.append([])
#                 for k in img_list1:
# #                     print(k)
#                     if 'com.png' in k:
#                         out_img_train.append(k)
#                     else:
#                         img.append(k)
        # # A
        # out_img_train = [lab for lab in img_list if 'GT.JPG' not in lab]
        # # B
        # gt_img_train = [lab for lab in img_list if 'com.JPG' in lab]
        # print(len(os.listdir(r'C:\Users\admin\Downloads\matlab_compose2')))
# #         print(img_list1)
#         print(len(img))
# #         print(len(out_img_train))
#         print(len(gt_img_train))
        # print((out_img_train[455:]))
        # print((gt_img_train[455:]))
        # print((img[455:]))
        # gt_img_train

        self.train = train
        self.gt_img_train = gt_img_train
        self.out_img_train = out_img_train
        self.img = img
        self.transforms = transforms
        # self.transforms_pyramid112 = T.Compose([T.Resize([112, 112]),T.ToTensor()])
        # self.transforms_pyramid61 = T.Compose([T.Resize([56,
        # 56]),T.ToTensor()])

    def __getitem__(self, index):
#         print(index)
#         print(self.out_img_train[index])
        augmentation = False
        filename = self.img[2 * index]
        label_path = self.gt_img_train[index]
#         out_path = self.out_img_train[index]
        img1_path = self.img[2 * index]
        img2_path = self.img[2 * index + 1]

        # print(label_path,self.out_img_train[index])

        label = Image.open(label_path).convert('YCbCr')
        # print(img1_path)
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img1_ycbcr = img1.convert('YCbCr')
        img2_ycbcr = img2.convert('YCbCr')
        img1_np = np.array(img1.resize((512, 512)))
        img2_np = np.array(img2.resize((512, 512)))

        if augmentation:
            img1_np = adjust_gamma(img1_np, gamma=random.uniform(0.5, 4))
            img2_np = adjust_gamma(img2_np, gamma=random.uniform(0.5, 4))
            # print("=>>>>>>>>>>>>>>>>>>>>>>>>>>>data augmentation on")
        # out_np = two_fusion(img1_np,img2_np)
        # raw_fused = np.uint8(np.clip(cfusion(raw0_new, raw1_new) * 255, 0,
        # 255))
        raw_fused = np.uint8(np.clip(cfusion(img1_np, img2_np) * 255, 0, 255))
        # if self.train==False:
        # imsave(".\\Result_check2\\%06s_compose.png" %
        #        (filename.split('\\')[-2]), img_as_ubyte(raw_fused))
        raw_fused_cbcr = rgb2ycbcr(raw_fused)[:, :, 0:3]
        # detail, luma = fusion(raw_fused, multi=True)

#         print(out_np)
#         print(out_np.shape)
#         print(img1_np.shape)
#         print(img2_np.shape)
#         print(out_np.shape)
#         print(out_np)

            # np.uint8 error in Image.fromarray
#         out = Image.fromarray(out_np.astype(np.uint8))
        # out = Image.fromarray(rgb2ycbcr(np.clip(out_np,0,255)).astype(np.uint8))
        # out = Image.fromarray(rgb2ycbcr(np.clip(out_np,0,255)).astype(np.uint8))
        # image = Image.fromarray((np.clip(out_np,0,255)).astype(np.uint8))

        image = Image.fromarray(raw_fused_cbcr.astype(np.uint8))
        # image = Image.fromarray(raw_fused_cbcr.astype(np.uint8))

        # ycbcr = image.convert('YCbCr')
        # B = np.ndarray((image.size[1], image.size[0], 3), 'u1', ycbcr.tobytes())
        # out = Image.fromarray(B[:,:,0], "L")
        out = image
        # out_ycbcr = rgb2ycbcr(out)
        # label_ycbcr = rgb2ycbcr(label)
#         out = Image.fromarray(out_np.astype(np.uint8))
#         print(out)
#         out = Image.open(out_path)

#         groundtruth256 = cv2.pyrDown(np.array(label))
#         print(groundtruth256.shape)
#         groundtruth128 = cv2.pyrDown(groundtruth256)
#         print(groundtruth128.shape)
#         groundtruth256 = Image.fromargit: commitray(groundtruth256)
#         groundtruth128 = Image.fromarray(groundtruth128)
        # filename = self.out_img_train[index][-16:]

        if self.transforms:
            img1 = self.transforms(img1_ycbcr)
            img2 = self.transforms(img2_ycbcr)
            out = self.transforms(out)
            label = self.transforms(label)
#             groundtruth256 = self.transforms_pyramid112(groundtruth256)
#             groundtruth128 = self.transforms_pyramid61(groundtruth128)
        return img1, img2, out, label, filename

    def __len__(self):

        return len(self.gt_img_train)


class HDRDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.folder = root
        self.indexes = open(root + '\annotations.txt').read().splitlines()

    def __getitem__(self, index):
        ldr_image, hdr_image = self.indexes[index].split('\t')
        ldr_image = cv2.imread(ldr_image)
        ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB)
        ldr_image = ldr_image / 255

        hdr_image = cv2.imread(hdr_image, cv2.IMREAD_ANYDEPTH)
        hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)

        return ldr_image.transpose(2, 0, 1), hdr_image.transpose(2, 0, 1)

    def __len__(self):
        return len(self.indexes)


def get_loader(root, batch_size, shuffle=True):
    # dataset = HDRDataset(root=root)

    # num_train = int(len(dataset) * 0.8)
    # num_val = len(dataset) - num_train
    # train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=shuffle,
    #                           drop_last=True)

    # val_loader = DataLoader(dataset=val_dataset,
    #                         batch_size=batch_size,
    #                         shuffle=shuffle,
    #                         drop_last=True)
    transforms = T.Compose([T.Resize([512, 512]), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
    # transforms = T.Compose([T.Resize([224, 224]), T.ToTensor(), torch.tensor([0.5, 0.5, 0.5]) ])
    # transforms = T.Compose([T.Resize([224, 224]),T.ToTensor(),
    # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset=HDRdatasets_dynamic_compose(True, transforms)
    test_dataset=HDRdatasets_dynamic_compose(False, transforms)
    train_dataloader=data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader=data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return train_dataloader, test_dataloader


    # return train_loader, val_loader

# if __name__ == '__main__':

#     transforms=T.Compose([T.Resize([512, 512]), T.ToTensor()])

# # #     test_mean, test_std = torch.tensor([0.5 ,0.5 ,0.5]), torch.tensor([0.5 ,0.5, 0.5])
#     train_dataset=HDRdatasets_dynamic_compose(False, transforms)
#     train_dataloader=data.DataLoader(
#         train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
# #     # for folder in train_dataloader:
#     for img1, img2, out, label, filename in train_dataloader:
#         print(filename[0].split('\\')[-2])
#         print(filename[0].replace('.JPG','').split('\\')[-1].split('\\')[0])
#         break
