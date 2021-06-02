# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 

'''
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import math
import numpy as np
from torch.autograd import Variable
import torch

import matplotlib
import sys
matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)


class ImageProcessing(object):

    @staticmethod
    def rgb_to_lab(img, is_training=True):
        """ PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor

        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)

        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.000001) + 0.055) / 1.055) ** 2.4) * img.gt(0.04045).float()

        rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169,
             0.950227],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(img, Variable(torch.FloatTensor(
            [1/0.950456, 1.0, 1/1.088754]), requires_grad=False).cuda())

        epsilon = 6/29

        img = ((img / (3.0 * epsilon**2) + 4.0/29.0) * img.le(epsilon**3).float()) + \
            (torch.clamp(img, min=0.0001) **
             (1.0/3.0) * img.gt(epsilon**3).float())

        fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0,  500.0,    0.0],  # fx
                                                    # fy
                                                    [116.0, -500.0,  200.0],
                                                    # fz
                                                    [0.0,    0.0, -200.0],
                                                    ]), requires_grad=False).cuda()

        img = torch.matmul(img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).cuda()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        '''
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        '''
        img[0, :, :] = img[0, :, :]/100
        img[1, :, :] = (img[1, :, :]/110 + 1)/2
        img[2, :, :] = (img[2, :, :]/110 + 1)/2

        img[(img != img).detach()] = 0

        img = img.contiguous()

        return img.cuda()

    @staticmethod
    def lab_to_rgb(img, is_training=True):
        """ PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img: image to be adjusted
        :returns: adjusted image
        :rtype: Tensor
        """                
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)
        img_copy = img.clone()

        img_copy[:, 0] = img[:, 0] * 100
        img_copy[:, 1] = ((img[:, 1] * 2)-1)*110
        img_copy[:, 2] = ((img[:, 2] * 2)-1)*110

        img = img_copy.clone().cuda()
        del img_copy

        lab_to_fxfyfz = Variable(torch.FloatTensor([  # X Y Z
            [1/116.0, 1/116.0, 1/116.0],  # R
            [1/500.0, 0, 0],  # G
            [0, 0, -1/200.0],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(
            img + Variable(torch.cuda.FloatTensor([16.0, 0.0, 0.0])), lab_to_fxfyfz)

        epsilon = 6.0/29.0

        img = (((3.0 * epsilon**2 * (img-4.0/29.0)) * img.le(epsilon).float()) +
               ((torch.clamp(img, min=0.0001)**3.0) * img.gt(epsilon).float()))

        # denormalize for D65 white point
        img = torch.mul(img, Variable(
            torch.cuda.FloatTensor([0.950456, 1.0, 1.088754])))

        xyz_to_rgb = Variable(torch.FloatTensor([  # X Y Z
            [3.2404542, -0.9692660,  0.0556434],  # R
            [-1.5371385,  1.8760108, -0.2040259],  # G
            [-0.4985314,  0.0415560,  1.0572252],  # B
        ]), requires_grad=False).cuda()

        img = torch.matmul(img, xyz_to_rgb)

        img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
                                                                        min=0.0001) ** (1/2.4) * 1.055) - 0.055) * img.gt(0.0031308).float()

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        img = img.contiguous()
        img[(img != img).detach()] = 0
        
        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        """Move the image channels to the first dimension of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def swapimdims_HW3_3HW(img):
        """Move the image channels to the last dimensiion of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    @staticmethod
    def load_image(img_filepath, normaliser):
        """Loads an image from file as a numpy multi-dimensional array

        :param img_filepath: filepath to the image
        :returns: image as a multi-dimensional numpy array
        :rtype: multi-dimensional numpy array

        """
        img = ImageProcessing.normalise_image(np.array(Image.open(img_filepath)), normaliser)  # NB: imread normalises to 0-1
        return img

    @staticmethod
    def normalise_image(img, normaliser):
        """Normalises image data to be a float between 0 and 1

        :param img: Image as a numpy multi-dimensional image array
        :returns: Normalised image as a numpy multi-dimensional image array
        :rtype: Numpy array

        """
        img = img.astype('float32') / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity):
        """Computes the PSNR for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for i in range(0, num_images):
            imageA = image_batchA[i, 0:3, :, :]
            imageB = image_batchB[i, 0:3, :, :]
            imageB = np.maximum(0, np.minimum(imageB, max_intensity))
            psnr_val += 10 * \
                np.log10(max_intensity ** 2 /
                         ImageProcessing.compute_mse(imageA, imageB))

        return psnr_val / num_images

    @staticmethod
    def compute_ssim(image_batchA, image_batchB):
        """Computes the SSIM for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        ssim_val = 0.0

        for i in range(0, num_images):
            imageA = ImageProcessing.swapimdims_3HW_HW3(
                image_batchA[i, 0:3, :, :])
            imageB = ImageProcessing.swapimdims_3HW_HW3(
                image_batchB[i, 0:3, :, :])
            ssim_val += ssim(imageA, imageB, data_range=imageA.max() - imageA.min(), multichannel=True,
                             gaussian_weights=True, win_size=11)

        return ssim_val / num_images

    @staticmethod
    def hsv_to_rgb(img):
        """Converts a HSV image to RGB
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: HSV image
        :returns: RGB image
        :rtype: Tensor

        """
        img=torch.clamp(img,0,1)
        img = img.permute(2, 1, 0)
        
        m1 = 0
        m2 = (img[:, :, 2]*(1-img[:, :, 1])-img[:, :, 2])/60
        m3 = 0
        m4 = -1*m2
        m5 = 0

        r = img[:, :, 2]+torch.clamp(img[:, :, 0]*360-0, 0, 60)*m1+torch.clamp(img[:, :, 0]*360-60, 0, 60)*m2+torch.clamp(
            img[:, :, 0]*360-120, 0, 120)*m3+torch.clamp(img[:, :, 0]*360-240, 0, 60)*m4+torch.clamp(img[:, :, 0]*360-300, 0, 60)*m5

        m1 = (img[:, :, 2]-img[:, :, 2]*(1-img[:, :, 1]))/60
        m2 = 0
        m3 = -1*m1
        m4 = 0

        g = img[:, :, 2]*(1-img[:, :, 1])+torch.clamp(img[:, :, 0]*360-0, 0, 60)*m1+torch.clamp(img[:, :, 0]*360-60,
                                                                                                0, 120)*m2+torch.clamp(img[:, :, 0]*360-180, 0, 60)*m3+torch.clamp(img[:, :, 0]*360-240, 0, 120)*m4

        m1 = 0
        m2 = (img[:, :, 2]-img[:, :, 2]*(1-img[:, :, 1]))/60
        m3 = 0
        m4 = -1*m2

        b = img[:, :, 2]*(1-img[:, :, 1])+torch.clamp(img[:, :, 0]*360-0, 0, 120)*m1+torch.clamp(img[:, :, 0]*360 -
                                                                                                 120, 0, 60)*m2+torch.clamp(img[:, :, 0]*360-180, 0, 120)*m3+torch.clamp(img[:, :, 0]*360-300, 0, 60)*m4

        img = torch.stack((r, g, b), 2)
        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()
        img = torch.clamp(img, 0, 1)

        return img



    @staticmethod
    def rgb_to_hsv(img):
        """Converts an RGB image to HSV
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: RGB image
        :returns: HSV image
        :rtype: Tensor

        """
        img=torch.clamp(img,1e-10,1)       

        img = img.permute(2, 1, 0)
        shape = img.shape

        img = img.contiguous()
        img = img.view(-1, 3)

        mx = torch.max(img, 1)[0]
        mn = torch.min(img, 1)[0]

        ones = Variable(torch.FloatTensor(
            torch.ones((img.shape[0])))).cuda()
        zero = Variable(torch.FloatTensor(torch.zeros(shape[0:2]))).cuda()

        img = img.view(shape)

        ones1 = ones[0:math.floor((ones.shape[0]/2))]
        ones2 = ones[math.floor(ones.shape[0]/2):(ones.shape[0])]

        mx1 = mx[0:math.floor((ones.shape[0]/2))]
        mx2 = mx[math.floor(ones.shape[0]/2):(ones.shape[0])]
        mn1 = mn[0:math.floor((ones.shape[0]/2))]
        mn2 = mn[math.floor(ones.shape[0]/2):(ones.shape[0])]

        df1 = torch.add(mx1, torch.mul(ones1*-1, mn1))
        df2 = torch.add(mx2, torch.mul(ones2*-1, mn2))

        df = torch.cat((df1, df2), 0)
        del df1, df2
        df = df.view(shape[0:2])+1e-10
        mx = mx.view(shape[0:2])

        img = img.cuda()
        df = df.cuda()
        mx = mx.cuda()

        g = img[:, :, 1].clone().cuda()
        b = img[:, :, 2].clone().cuda()
        r = img[:, :, 0].clone().cuda()

        img_copy = img.clone()
        
        img_copy[:, :, 0] = (((g-b)/df)*r.eq(mx).float() + (2.0+(b-r)/df)
                         * g.eq(mx).float() + (4.0+(r-g)/df)*b.eq(mx).float())
        img_copy[:, :, 0] = img_copy[:, :, 0]*60.0

        zero = zero.cuda()
        img_copy2 = img_copy.clone()

        img_copy2[:, :, 0] = img_copy[:, :, 0].lt(zero).float(
        )*(img_copy[:, :, 0]+360) + img_copy[:, :, 0].ge(zero).float()*(img_copy[:, :, 0])

        img_copy2[:, :, 0] = img_copy2[:, :, 0]/360

        del img, r, g, b

        img_copy2[:, :, 1] = mx.ne(zero).float()*(df/mx) + \
            mx.eq(zero).float()*(zero)
        img_copy2[:, :, 2] = mx
        
        img_copy2[(img_copy2 != img_copy2).detach()] = 0

        img = img_copy2.clone()

        img = img.permute(2, 1, 0)
        img = torch.clamp(img, 1e-10, 1)

        return img

    
    @staticmethod
    def apply_curve(img, C, slope_sqr_diff, channel_in, channel_out,
    clamp=False):
        """Applies a peicewise linear curve defined by a set of knot points to
        an image channel

        :param img: image to be adjusted
        :param C: predicted knot points of curve
        :returns: adjusted image
        :rtype: Tensor

        """
        slope = Variable(torch.zeros((C.shape[0]-1))).cuda()
        curve_steps = C.shape[0]-1

        '''
        Compute the slope of the line segments
        '''
        for i in range(0, C.shape[0]-1):
            slope[i] = C[i+1]-C[i]

        '''
        Compute the squared difference between slopes
        '''
        for i in range(0, slope.shape[0]-1):
            slope_sqr_diff += (slope[i+1]-slope[i])*(slope[i+1]-slope[i])

        '''
        Use predicted line segments to compute scaling factors for the channel
        '''
        scale = float(C[0])
        for i in range(0, slope.shape[0]-1):
            if clamp:
                scale += float(slope[i])*(torch.clamp(img[:, :,channel_in]*curve_steps-i,0,1))
            else:
                scale += float(slope[i])*(img[:, :,channel_in]*curve_steps-i)
                
        img_copy = img.clone()
  
        img_copy[:, :, channel_out] = img[:, :, channel_out]*scale
        
        img_copy = torch.clamp(img_copy,0,1)
        
        return img_copy, slope_sqr_diff

    @staticmethod
    def adjust_hsv(img, S):
        """Adjust the HSV channels of a HSV image using learnt curves

        :param img: image to be adjusted 
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.squeeze(0).permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        S1 = torch.exp(S[0:int(S.shape[0]/4)])
        S2 = torch.exp(S[(int(S.shape[0]/4)):(int(S.shape[0]/4)*2)])
        S3 = torch.exp(S[(int(S.shape[0]/4)*2):(int(S.shape[0]/4)*3)])
        S4 = torch.exp(S[(int(S.shape[0]/4)*3):(int(S.shape[0]/4)*4)])

        slope_sqr_diff = Variable(torch.zeros(1)*0.0).cuda()

        '''
        Adjust Hue channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, S1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Adjust Saturation channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1)
        
        '''
        Adjust Saturation channel based on Saturation using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Adjust Value channel based on Value using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()
        
        return img, slope_sqr_diff

    @staticmethod
    def adjust_rgb(img, R):
        """Adjust the RGB channels of a RGB image using learnt curves

        :param img: image to be adjusted 
        :param S: predicted parameters of piecewise linear curves
        :returns: adjust image, regularisation term
        :rtype: Tensor, float

        """
        img = img.squeeze(0).permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()

        '''
        Extract the parameters of the three curves
        '''
        R1 = torch.exp(R[0:int(R.shape[0]/3)])
        R2 = torch.exp(R[(int(R.shape[0]/3)):(int(R.shape[0]/3)*2)])
        R3 = torch.exp(R[(int(R.shape[0]/3)*2):(int(R.shape[0]/3)*3)])

        '''
        Apply the curve to the R channel 
        '''
        slope_sqr_diff = Variable(torch.zeros(1)*0.0).cuda()

        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, R1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Apply the curve to the G channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Apply the curve to the B channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R3, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff

    @staticmethod
    def adjust_lab(img, L):
        """Adjusts the image in LAB space using the predicted curves

        :param img: Image tensor
        :param L: Predicited curve parameters for LAB channels
        :returns: adjust image, and regularisation parameter
        :rtype: Tensor, float

        """
        img = img.permute(2, 1, 0)

        shape = img.shape
        img = img.contiguous()

        '''
        Extract predicted parameters for each L,a,b curve
        '''
        L1 = torch.exp(L[0:int(L.shape[0]/3)])
        L2 = torch.exp(L[(int(L.shape[0]/3)):(int(L.shape[0]/3)*2)])
        L3 = torch.exp(L[(int(L.shape[0]/3)*2):(int(L.shape[0]/3)*3)])

        slope_sqr_diff = Variable(torch.zeros(1)*0.0).cuda()

        '''
        Apply the curve to the L channel 
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, L1, slope_sqr_diff, channel_in=0, channel_out=0)

        '''
        Now do the same for the a channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1)

        '''
        Now do the same for the b channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(2, 1, 0)
        img = img.contiguous()

        return img, slope_sqr_diff
