# -*- coding: utf-8 -*-

from skimage import io
from skimage.color import label2rgb, separate_stains
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_opening, binary_closing, opening
from skimage.future.graph import ncut, rag_mean_color
from skimage.segmentation import clear_border, find_boundaries, mark_boundaries, quickshift, slic

from scipy.ndimage.measurements import find_objects
from scipy import ndimage as ndi

from sklearn import cluster

from PIL import Image

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.set_printoptions(threshold=np.nan)

cluster_num = 3

conv_matrix = np.array([[0.644211, 0.716556, 0.266844],
                        [0.092789, 0.954111, 0.283111],
                        [0.000000, 0.000000, 0.000000]])


def rgb2gray(rgb):

    return 0.2125 * rgb[0] + 0.7154 * rgb[1] + 0.0721 * rgb[2]


def rgb2rgba(img):

    img_pil_rgb = Image.fromarray(img)

    img_pil_rgba = img_pil_rgb.convert("RGBA")

    img_rgba = np.asarray(img_pil_rgba)

    img_rgba.setflags(write=1)

    return img_rgba


path = sys.argv[1]

image = io.imread(path)

image = image[:, :, :3]

w, h, d = image.shape

pixels = np.reshape(image, (w * h, d))

kmeans = cluster.KMeans(n_clusters=cluster_num)

kmeans.fit(pixels)

labels = kmeans.labels_.reshape((w, h))

image_ = np.copy(image)

nuclei = np.zeros((w, h))

graies = [(rgb2gray(np.average(image[labels == i], axis=0)), i) for i in xrange(cluster_num)]

graies.sort()

nuclei[labels == graies[0][1]] = 1

image_[labels == graies[1][1]] = 255

image_[labels == graies[2][1]] = 255

hematoxylin = separate_stains(image_, conv_matrix)[:, :, 0]

opening = opening(hematoxylin, disk(int(sys.argv[2])))

labeled = label(opening)

boundaries = find_boundaries(opening, mode="inner")

ax1 = plt.subplot2grid((2, 4), (0, 0))

ax1.imshow(image)

ax2 = plt.subplot2grid((2, 4), (1, 0))

ax2.imshow(nuclei)

ax3 = plt.subplot2grid((2, 4), (0, 1))

ax3.imshow(image_)

ax4 = plt.subplot2grid((2, 4), (1, 1))

ax4.imshow(hematoxylin)

ax5 = plt.subplot2grid((2, 4), (0, 2))

ax5.imshow(hematoxylin)

ax6 = plt.subplot2grid((2, 4), (1, 2))

ax6.imshow(opening)

ax7 = plt.subplot2grid((2, 4), (0, 3))

ax7.imshow(label2rgb(labeled, image=image))

for region in regionprops(labeled):

    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=0, edgecolor='red', linewidth=2)

    ax7.add_patch(rect)

ax8 = plt.subplot2grid((2, 4), (1, 3))

ax8.imshow(mark_boundaries(image, boundaries, color=(1, 0, 0)))

plt.show()
