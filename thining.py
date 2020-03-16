#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageFilter
from math import sqrt
import imageio
from skimage.io import imread
import numpy as np
sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

def merge_images(a, b, f):
    result = a.copy()
    result_load = result.load()
    a_load = a.load()
    b_load = b.load()

    (x, y) = a.size
    for i in range(0, x):
        for j in range(0, y):
            result_load[i, j] = f(a_load[i, j], b_load[i, j])

    return result

def partial_sobels(im):
    ySobel =tuple( im.filter(ImageFilter.Kernel((3, 3), flatten(sobelOperator), 1)))
    xSobel = tuple(im.filter(ImageFilter.Kernel((3, 3), flatten(transpose(sobelOperator)), 1)))
    return (xSobel, ySobel)

def full_sobels(im):
    (xSobel, ySobel) = partial_sobels(im)
    sobel = merge_images(xSobel, ySobel, lambda x, y: sqrt(x**2 + y**2))
    return (xSobel, ySobel, sobel)


from PIL import Image, ImageDraw
import math
import copy

def apply_kernel_at(get_value, kernel, i, j):
    kernel_size = len(kernel)
    result = 0
    for k in range(0, kernel_size):
        for l in range(0, kernel_size):
            pixel = get_value(i + k - kernel_size / 2, j + l - kernel_size / 2)
            result += pixel * kernel[k][l]
    return result

def apply_to_each_pixel(pixels, f):
    for i in range(0, len(pixels)):
        for j in range(0, len(pixels[i])):
            pixels[i][j] = f(pixels[i][j])

def calculate_angles(im, W, f, g):
    (x, y) = im.size
    im_load = im.load()
    get_pixel = lambda x, y: im_load[x, y]

    ySobel = sobelOperator
    xSobel = transpose(sobelOperator)

    result = [[] for i in range(1, x, W)]

    for i in range(1, x, W):
        for j in range(1, y, W):
            nominator = 0
            denominator = 0
            for k in range(i, min(i + W , x - 1)):
                for l in range(j, min(j + W, y - 1)):
                    Gx = apply_kernel_at(get_pixel, xSobel, k, l)
                    Gy = apply_kernel_at(get_pixel, ySobel, k, l)
                    nominator += f(Gx, Gy)
                    denominator += g(Gx, Gy)
            angle = (math.pi + math.atan2(nominator, denominator)) / 2
            result[(i - 1) / W].append(angle)

    return result

def flatten(ls):
    return reduce(lambda x, y: x + y, ls, [])

def transpose(ls):
    return map(list, zip(*ls))

def gauss(x, y):
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))

def kernel_from_function(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel

def gauss_kernel(size):
    return kernel_from_function(size, gauss)

def apply_kernel(pixels, kernel):
    apply_kernel_with_f(pixels, kernel, lambda old, new: new)

def apply_kernel_with_f(pixels, kernel, f):
    size = len(kernel)
    for i in range(size / 2, len(pixels) - size / 2):
        for j in range(size / 2, len(pixels[i]) - size / 2):
            pixels[i][j] = f(pixels[i][j], apply_kernel_at(lambda x, y: pixels[x][y], kernel, i, j))

def smooth_angles(angles):
    cos_angles = copy.deepcopy(angles)
    sin_angles = copy.deepcopy(angles)
    apply_to_each_pixel(cos_angles, lambda x: math.cos(2 * x))
    apply_to_each_pixel(sin_angles, lambda x: math.sin(2 * x))

    kernel = gauss_kernel(5)
    apply_kernel(cos_angles, kernel)
    apply_kernel(sin_angles, kernel)

    for i in range(0, len(cos_angles)):
        for j in range(0, len(cos_angles[i])):
            cos_angles[i][j] = (math.atan2(sin_angles[i][j], cos_angles[i][j])) / 2

    return cos_angles
def load_image(im):
    (x, y) = im.size
    im_load = im.load()

    result = []
    for i in range(0, x):
        result.append([])
        for j in range(0, y):
            result[i].append(im_load[i, j])

    return result

def load_pixels(im, pixels):
    (x, y) = im.size
    im_load = im.load()

    for i in range(0, x):
        for j in range(0, y):
            im_load[i, j]= int(pixels[i][j])

def get_line_ends(i, j, W, tang):
    if -1 <= tang and tang <= 1:
        begin = (i, (-W/2) * tang + j + W/2)
        end = (i + W, (W/2) * tang + j + W/2)
    else:
        begin = (i + W/2 + W/(2 * tang), j + W/2)
        end = (i + W/2 - W/(2 * tang), j - W/2)
    return (begin, end)

def draw_lines(im, angles, W):
    (x, y) = im.size
    result = im.convert("RGB")

    draw = ImageDraw.Draw(result)

    for i in range(1, x, W):
        for j in range(1, y, W):
            tang = math.tan(angles[(i - 1) / W][(j - 1) / W])

            (begin, end) = get_line_ends(i, j, W, tang)
            draw.line([begin, end], fill=150)

    del draw

    return result

from PIL import Image, ImageDraw
import math
import os
from utils import flatten, transpose

usage = False

def apply_structure(pixels, structure, result):
    global usage
    usage = False

    def choose(old, new):
        global usage
        if new == result:
            usage = True
            return 0.0
        return old

    apply_kernel_with_f(pixels, structure, choose)

    return usage

def apply_all_structures(pixels, structures):
    usage = False
    for structure in structures:
        usage |= apply_structure(pixels, structure, flatten(structure).count(1))

    return usage

def make_thin(im):
    loaded = load_image(im)
    apply_to_each_pixel(loaded, lambda x: 0.0 if x > 10 else 1.0)
    print "loading phase done"

    t1 = [[1, 1, 1], [0, 1, 0], [0.1, 0.1, 0.1]]
    t2 = transpose(t1)
    t3 = reverse(t1)
    t4 = transpose(t3)
    t5 = [[0, 1, 0], [0.1, 1, 1], [0.1, 0.1, 0]]
    t7 = transpose(t5)
    t6 = reverse(t7)
    t8 = reverse(t5)

    thinners = [t1, t2, t3, t4, t5, t6, t7]

    usage = True
    while(usage):
        usage = apply_all_structures(loaded, thinners)
        print "single thining phase done"

    print "thining done"

    apply_to_each_pixel(loaded, lambda x: 255.0 * (1 - x))
    load_pixels(im, loaded)
    im.show()

def reverse(ls):
    cpy = ls[:]
    cpy.reverse()
    return cpy


    #parser = argparse.ArgumentParser(description="Image thining")
    #parser.add_argument("image", nargs=1, help = "Path to image")
    #parser.add_argument("--save", action='store_true', help = "Save result image as src_image_thinned.gif")
    #args = parser.parse_args()

im = Image.open("/home/priyanka/Desktop/output images 1/4.1.bmp")
im = im.convert("L")  # covert to grayscale
make_thin(im)
im=np.array(im)
imageio.imwrite("/home/priyanka/Desktop/output images 1/4.2.bmp", im)
    #if args.save:
        #base_image_name = os.path.splitext(os.path.basename(args.image[0]))[0]
        #im.save(base_image_name + "_thinned.gif", "GIF")


# In[10]:


from __future__ import division
import cv2
#import cv
import os
import sys
#import argparse
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
from numpy.lib import pad

def read_image(image_name):
# Load a subject's right hand, second finger image and risize to 352*352

    fingerprint = cv2.imread(image_name, 0)
    fingerprint = cv2.resize(fingerprint,(352,352))
    fpcopy = fingerprint[:]
    row, col = fingerprint.shape

    return row, col, fingerprint, fpcopy


def segment(r, c, finger_print, fp_copy):
# Image segmentation based on variance and mathematical morphology

    W = 16
    threshold = 1600
    A = np.zeros((r,c), np.uint8)

    for i in np.arange(0,r-1,W):
        for j in np.arange(0,c-1,W):
            Mw = (1/(W*W)) * (sum(finger_print[i:i+W,j:j+W]))
            Vw = (1/(W*W)) * (sum((finger_print[i:i+W,j:j+W] - Mw)**2))
            
            if (Vw < threshold).all():
                finger_print[i:i+W,j:j+W] = 0
                A[i:i+W,j:j+W] = 0
            else:
                A[i:i+W,j:j+W] = 1

    kernel = np.ones((44,44),np.uint8)
    closing_mask = cv2.morphologyEx(A, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((88,88),np.uint8)
    opening_mask = cv2.morphologyEx(closing_mask, cv2.MORPH_OPEN, kernel)

    for i in np.arange(0,r-1,W):
        for j in np.arange(0,c-1,W):
            if ((sum(finger_print[i:i+W,j:j+W])) != (sum(opening_mask[i:i+W,j:j+W]))).all():
                if np.mean(opening_mask[i:i+W,j:j+W]) == 1:
                    finger_print[i:i+W,j:j+W] = fp_copy[i:i+W,j:j+W]
                elif  np.mean(opening_mask[i:i+W,j:j+W]) == 0:
                    finger_print[i:i+W,j:j+W] = 0

    return finger_print, opening_mask


def normal(fingerprint, r, c):
# image normalization

    M0 = 100
    VAR0 = 1000
    M = np.mean(fingerprint[:])
    VAR = np.var(fingerprint[:])
    normalization = np.zeros((r,c), np.uint8)

    for i in range(r):
        for j in range(c):
            if (fingerprint[i,j] > M):
                normalization[i,j] = M0 + np.sqrt(VAR0 * ((fingerprint[i,j] - M)**2) / VAR)
            else:
                normalization[i,j] = M0 - np.sqrt(VAR0 * ((fingerprint[i,j] - M)**2) / VAR)

    return normalization

def removedot(invertThin):
# remove dots
    
    temp0 = np.array(invertThin[:])
    temp0 = np.array(temp0)
    temp1 = temp0/255
    temp2 = np.array(temp1)
    temp3 = np.array(temp2)
    
    enhanced_img = np.array(temp0)
    filter0 = np.zeros((10,10))
    W,H = temp0.shape[:2]
    filtersize = 6
    
    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize,j:j + filtersize]

            flag = 0
            if sum(filter0[:,0]) == 0:
                flag +=1
            if sum(filter0[:,filtersize - 1]) == 0:
                flag +=1
            if sum(filter0[0,:]) == 0:
                flag +=1
            if sum(filter0[filtersize - 1,:]) == 0:
                flag +=1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize))

    return temp2


def cross_number(enhanced_img, m, n):
# minutiae extraction using crossing number method

    r=0
    g=0
    row_start = 3
    col_start = 3
    mep = np.zeros((m,2))  # array for indices of minutiae points (end point)
    mbp = np.zeros((m,2))  # bifurcation point

    for i in range(row_start, m):
        for j in range(col_start, n):
            if enhanced_img[int(i),int(j)] == 1:
                cn = (1/2)*(abs(enhanced_img[i,j+1] - enhanced_img[i-1,j+1]) + abs(enhanced_img[i-1,j+1] - enhanced_img[i-1,j]) + abs(enhanced_img[i-1,j] - enhanced_img[i-1,j-1]) + abs(enhanced_img[i-1,j-1] - enhanced_img[i,j-1])+ abs(enhanced_img[i,j-1] - enhanced_img[i+1,j-1]) + abs(enhanced_img[i+1,j-1] - enhanced_img[i+1,j])+ abs(enhanced_img[i+1,j] - enhanced_img[i+1,j+1]) + abs(enhanced_img[i+1,j+1] - enhanced_img[i,j+1]))
                if cn == 1:
                    r = r+1
                    mep[r,:] = [i,j]
                elif cn == 3:
                    g = g+1
                    mbp[g,:] = [i,j]

    return mep, mbp


def marking_init(enhanced_img, mep, mbp):
# mark initially extracted minutiae points

    img_thin = np.array(enhanced_img[:])  # convert image to array for marking points

    fig = plt.figure(figsize=(10,8),dpi=30000)

    num1 = len(mep)
    num2 = len(mbp)

    figure, imshow(img_thin, cmap = cm.Greys_r)
    title('mark extracted points')
    plt.hold(True)

    for i in range(num1):
        xy = mep[i,:]
        u = xy[0]
        v = xy[1]
        if (u != 0.0) & (v != 0.0):
            plt.plot(v, u, 'r.', markersize = 7)

    plt.hold(True)

    for i in range(num2):
        xy = mbp[i,:]
        u = xy[0]
        v = xy[1]
        if (u != 0.0) & (v != 0.0):
            plt.plot(v, u, 'c+', markersize = 7)

    plt.show()
    cv2.imwrite("/home/priyanka/Desktop/initial_extraction.png", img_thin)
image_name= "/home/priyanka/Desktop/o_7.png"
row, col, fingerprint, fpcopy = read_image(image_name)
finger_print, opening_mask = segment(row, col, fingerprint, fpcopy)
    
dedot_image ="/home/priyanka/Desktop/o_7.png"  
end_point, bifur_point = cross_number(dedot_image, row, col)
marking_init(de_dot, end_point, bifur_point)


# In[ ]:




