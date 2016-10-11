# -*- coding: utf-8 -*-

from skimage import io
from skimage.color import label2rgb, separate_stains
from skimage.filters import rank
from skimage.measure import label, regionprops
from skimage.feature import blob_doh
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.morphology import disk, binary_opening, binary_closing, opening, closing
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

image = rank.equalize(image, disk(10))

w, h, d = image.shape

pixels = np.reshape(image, (w * h, d))


kmeans = cluster.KMeans(n_clusters=cluster_num)

kmeans.fit(pixels)

labels = kmeans.labels_.reshape((w, h))

graies = [(rgb2gray(np.average(image[labels == i], axis=0)), i) for i in xrange(cluster_num)]

graies.sort()

nuclei = np.zeros((w, h))

nuclei[labels == graies[0][1]] = 1

image_ = np.copy(image)

image_[labels == graies[1][1]] = 255

image_[labels == graies[2][1]] = 255

pixels = np.reshape(image_, (w * h, d))


kmeans = cluster.KMeans(n_clusters=cluster_num)

kmeans.fit(pixels)

labels = kmeans.labels_.reshape((w, h))

graies = [(rgb2gray(np.average(image[labels == i], axis=0)), i) for i in xrange(cluster_num)]

graies.sort()

nuclei_ = np.zeros((w, h))

nuclei_[labels == graies[0][1]] = 1

image__ = np.copy(image)

image__[labels == graies[1][1]] = 255

image__[labels == graies[2][1]] = 255

pixels = np.reshape(image__, (w * h, d))


kmeans = cluster.KMeans(n_clusters=cluster_num)

kmeans.fit(pixels)

labels = kmeans.labels_.reshape((w, h))

graies = [(rgb2gray(np.average(image[labels == i], axis=0)), i) for i in xrange(cluster_num)]

graies.sort()

nuclei__ = np.zeros((w, h))

nuclei__[labels == graies[0][1]] = 1

image___ = np.copy(image)

image___[labels == graies[1][1]] = 255

image___[labels == graies[2][1]] = 255


hematoxylin_ = separate_stains(image_, conv_matrix)[:, :, 0]

hematoxylin__ = separate_stains(image__, conv_matrix)[:, :, 0]

hematoxylin___ = separate_stains(image___, conv_matrix)[:, :, 0]

# opened_ = opening(hematoxylin_, disk(int(sys.argv[2])))
#
# opened__ = opening(hematoxylin__, disk(int(sys.argv[2])))

cleared = clear_border(nuclei)

cleared_ = clear_border(nuclei_)

cleared__ = clear_border(nuclei__)

# closed = binary_closing(cleared, disk(int(sys.argv[2])))
#
# closed_ = binary_closing(cleared_, disk(int(sys.argv[2])))
#
# closed__ = binary_closing(cleared__, disk(int(sys.argv[2])))

filled = ndi.binary_fill_holes(cleared)

filled_ = ndi.binary_fill_holes(cleared_)

filled__ = ndi.binary_fill_holes(cleared__)

bopened = binary_opening(filled, disk(int(sys.argv[2])))

bopened_ = binary_opening(filled_, disk(int(sys.argv[3])))

bopened__ = binary_opening(filled__, disk(int(sys.argv[4])))

boundaries = find_boundaries(bopened, mode="inner")

boundaries_ = find_boundaries(bopened_, mode="inner")

boundaries__ = find_boundaries(bopened__, mode="inner")

# plt.subplot(131)
#
# plt.imshow(hematoxylin_)
#
# plt.subplot(132)
#
# plt.imshow(hematoxylin__)
#
# plt.subplot(133)
#
# plt.imshow(hematoxylin___)
#
# plt.subplot(324)
#
# plt.imshow(mark_boundaries(image, boundaries_))
#
# plt.subplot(325)
#
# plt.imshow(nuclei__)
#
# plt.subplot(326)
#
# plt.imshow(mark_boundaries(image, boundaries__))

ax1 = plt.subplot2grid((3, 2), (0, 0))

ax1.imshow(nuclei)

ax2 = plt.subplot2grid((3, 2), (0, 1))

ax2.imshow(mark_boundaries(image, boundaries))

for region in regionprops(label(bopened)):

    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=0, edgecolor='red', linewidth=0.5)

    ax2.add_patch(rect)

ax3 = plt.subplot2grid((3, 2), (1, 0))

ax3.imshow(nuclei_)

ax4 = plt.subplot2grid((3, 2), (1, 1))

ax4.imshow(mark_boundaries(image, boundaries_))

for region in regionprops(label(bopened_)):

    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=0, edgecolor='red', linewidth=0.5)

    ax4.add_patch(rect)

ax5 = plt.subplot2grid((3, 2), (2, 0))

ax5.imshow(nuclei__)

ax6 = plt.subplot2grid((3, 2), (2, 1))

ax6.imshow(mark_boundaries(image, boundaries__))

for region in regionprops(label(bopened__)):

    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=0, edgecolor='red', linewidth=0.5)

    ax6.add_patch(rect)

# ax7 = plt.subplot2grid((2, 4), (0, 3))
#
# ax7.imshow(label2rgb(labeled, image=image))
#
# for region in regionprops(labeled):
#
#     minr, minc, maxr, maxc = region.bbox
#
#     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=0, edgecolor='red', linewidth=2)
#
#     ax7.add_patch(rect)
#
# ax8 = plt.subplot2grid((2, 4), (1, 3))
#
# ax8.imshow(mark_boundaries(image, boundaries, color=(0, 1, 0)))

plt.show()
