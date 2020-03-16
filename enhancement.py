#!/usr/bin/env python
# coding: utf-8

# In[2]:


from PIL import Image, ImageDraw
import math
from PIL import Image, ImageFilter
from math import sqrt
import utils

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
    xSobel = tuple(im.filter(ImageFilter.Kernel((3, 3), flatten(utils.transpose(sobelOperator)), 1)))
    return (xSobel, ySobel)

def full_sobels(im):
    (xSobel, ySobel) = partial_sobels(im)
    sobel = merge_images(xSobel, ySobel, lambda x, y: sqrt(x**2 + y**2))
    return (xSobel, ySobel, sobel)





from PIL import Image, ImageDraw
import math
import sobel
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
            im_load[i, j] = int(pixels[i][j])

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
import utils
import argparse
import math


def points_on_line(line, W):
    im = Image.new("L", (W, 3 * W), 100)
    draw = ImageDraw.Draw(im)
    draw.line([(0, line(0) + W), (W, line(W) + W)], fill=10)
    im_load = im.load()

    points = []
    for x in range(0, W):
        for y in range(0, 3 * W):
            if im_load[x, y] == 10:
               points.append((x, y - W))

    del draw
    del im

    dist = lambda (x, y): (x - W / 2) ** 2 + (y - W / 2) ** 2

    return sorted(points, cmp = lambda x, y: dist(x) < dist(y))[:W]

def vec_and_step(tang, W):
    (begin, end) = get_line_ends(0, 0, W, tang)
    (x_vec, y_vec) = (end[0] - begin[0], end[1] - begin[1])
    length = math.hypot(x_vec, y_vec)
    (x_norm, y_norm) = (x_vec / length, y_vec / length)
    step = length / W

    return (x_norm, y_norm, step)

def block_frequency(i, j, W, angle, im_load):
    tang = math.tan(angle)
    ortho_tang = -1 / tang

    (x_norm, y_norm, step) = vec_and_step(tang, W)
    (x_corner, y_corner) = (0 if x_norm >= 0 else W, 0 if y_norm >= 0 else W)

    grey_levels = []

    for k in range(0, W):
        line = lambda x: (x - x_norm * k * step - x_corner) * ortho_tang + y_norm * k * step + y_corner
        points = points_on_line(line, W)
        level = 0
        for point in points:
            level += im_load[point[0] + i * W, point[1] + j * W]
        grey_levels.append(level)

    treshold = 100
    upward = False
    last_level = 0
    last_bottom = 0
    count = 0.0
    spaces = len(grey_levels)
    for level in grey_levels:
        if level < last_bottom:
            last_bottom = level
        if upward and level < last_level:
            upward = False
            if last_bottom + treshold < last_level:
                count += 1
                last_bottom = last_level
        if level > last_level:
            upward = True
        last_level = level

    return count / spaces if spaces > 0 else 0

def freq(im, W, angles):
    (x, y) = im.size
    im_load = im.load()
    freqs = [[0] for i in range(0, x / W)]

    for i in range(1, x / W - 1):
        for j in range(1, y / W - 1):
            freq = block_frequency(i, j, W, angles[i][j], im_load)
            freqs[i].append(freq)
        freqs[i].append(0)

    freqs[0] = freqs[-1] = [0 for i in range(0, y / W)]

    return freqs

def freq_img(im, W, angles):
    (x, y) = im.size
    freqs = freq(im, W, angles)
    freq_img = im.copy()

    for i in range(1, x / W - 1):
        for j in range(1, y / W - 1):
            box = (i * W, j * W, min(i * W + W, x), min(j * W + W, y))
            freq_img.paste(int(freqs[i][j] * 255.0 * 1.2),box)

    return freq_img

W = int(18)

f = lambda x, y: 2 * x * y
g = lambda x, y: x ** 2 - y ** 2
im = Image.open("/home/priyanka/Desktop/biometrics project/database/Tsinghua Distorted Fingerprint Database/1_2.bmp")
im = im.convert("L")
angles = calculate_angles(im, W, f, g)
if True:
    angles = smooth_angles(angles)

#freq_img = freq_img(im, W, angles)
#freq_img.show()


im = Image.open("/home/priyanka/Desktop/output images/ppf1.png")
im = im.convert("L")  # covert to grayscale
#im.show()

W = int(18)

f = lambda x, y: 2 * x * y
g = lambda x, y: x ** 2 - y ** 2

angles = calculate_angles(im, W, f, g)
if True:
    angles = smooth_angles(angles)

#freq_img = freq_img(im, W, angles)
#im=np.array(im)
#freq_img.show()

from PIL import Image, ImageDraw
import utils
import argparse
import math
import frequency
import os

def gabor_kernel(W, angle, freq):
    cos = math.cos(angle)
    sin = math.sin(angle)

    yangle = lambda x, y: x * cos + y * sin
    xangle = lambda x, y: -x * sin + y * cos

    xsigma = ysigma = 4

    return kernel_from_function(W, lambda x, y:
        math.exp(-(
            (xangle(x, y) ** 2) / (xsigma ** 2) +
            (yangle(x, y) ** 2) / (ysigma ** 2)) / 2) *
        math.cos(2 * math.pi * freq * xangle(x, y)))

def gabor(im, W, angles):
    (x, y) = im.size
    im_load = im.load()

    freqs = freq(im, W, angles)
    print "computing local ridge frequency done"

    gauss = gauss_kernel(3)
    apply_kernel(freqs, gauss)

    for i in range(1, x / W - 1):
        for j in range(1, y / W - 1):
            kernel = gabor_kernel(W, angles[i][j], freqs[i][j])
            for k in range(0, W):
                for l in range(0, W):
                    im_load[i * W + k, j * W + l] = int(apply_kernel_at(
                        lambda x, y: im_load[x, y],
                        kernel,
                        i * W + k,
                        j * W + l))

    return im


im = Image.open("/home/priyanka/Desktop/biometrics project/database/Tsinghua Distorted Fingerprint Database/1_2.bmp")
im = im.convert("L")  # covert to grayscale
#im.show()

W = int(18)

f = lambda x, y: 2 * x * y
g = lambda x, y: x ** 2 - y ** 2

angles = calculate_angles(im, W, f, g)
print "calculating orientation done"

angles = smooth_angles(angles)
print "smoothing angles done"

result = gabor(im, W, angles)
result.show()
result.save("/home/priyanka/Desktop/output images 1/4.1.bmp")

    


# In[ ]:





# In[ ]:




