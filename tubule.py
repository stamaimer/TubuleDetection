# -*- coding: utf-8 -*-

from skimage import io
from skimage.color import label2rgb
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import disk, watershed, binary_opening, binary_closing
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

tubule = np.zeros((w, h))

nuclei = np.zeros((w, h))

graies = [(rgb2gray(np.average(image[labels == i], axis=0)), i) for i in xrange(cluster_num)]

graies.sort()

nuclei[labels == graies[0][1]] = 1

# tubule[labels == graies[1][1]] = 1

tubule[labels == graies[-1][1]] = 1
#
# image_[labels == graies[-1][1]] = 255
#
# image_[labels == graies[1][1]] = 0
#
# image_[labels == graies[0][1]] = 0

# distance = ndi.distance_transform_edt(tubule)
#
# local_maxi = peak_local_max(distance, indices=0, footprint=np.ones((3, 3)), labels=tubule)
#
# markers = ndi.label(local_maxi)[0]
#
# labels = watershed(-distance, markers, mask=tubule)

cleared = clear_border(tubule)

closed = binary_closing(cleared, disk(int(sys.argv[2])))

filled = ndi.binary_fill_holes(closed)

opened = binary_opening(filled, disk(int(sys.argv[3])))

labeled = label(opened)

boundaries = find_boundaries(labeled, mode="inner")

ax1 = plt.subplot2grid((2, 4), (0, 0))

ax1.imshow(image)

ax2 = plt.subplot2grid((2, 4), (1, 0))

ax2.imshow(tubule)

ax3 = plt.subplot2grid((2, 4), (0, 1))

ax3.imshow(cleared)

ax4 = plt.subplot2grid((2, 4), (1, 1))

ax4.imshow(closed)

ax5 = plt.subplot2grid((2, 4), (0, 2))

ax5.imshow(filled)

ax6 = plt.subplot2grid((2, 4), (1, 2))

ax6.imshow(opened)

ax7 = plt.subplot2grid((2, 4), (0, 3))

ax7.imshow(label2rgb(labeled, image=image))

for region in regionprops(labeled):

    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=0, edgecolor='red', linewidth=2)

    ax7.add_patch(rect)

ax8 = plt.subplot2grid((2, 4), (1, 3))

ax8.imshow(mark_boundaries(image, boundaries, color=(1, 0, 0)))

plt.show()

# _, axes = plt.subplots(2, 3)
#
# axes[0][0].imshow(image)
#
# axes[1][0].imshow(tubule)
#
# axes[0][1].imshow(cleared)
#
# axes[1][1].imshow(closed)
#
# axes[0][2].imshow(filled)
#
# axes[1][2].imshow(label2rgb(labeled, image=image))
#
# for region in regionprops(labeled):
#
#     minr, minc, maxr, maxc = region.bbox
#
#     rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=0, edgecolor='red', linewidth=2)
#
#     axes[1][2].add_patch(rect)
#
# plt.show()

# kernel_size = cluster.estimate_bandwidth(np.reshape(image_, (w * h, d)), n_samples=1000)
#
# print kernel_size
#
# segments = quickshift(image_, ratio=0.5, kernel_size=kernel_size)
#
# # segments = slic(image)
#
# print len(np.unique(segments))
#
# plt.subplot(1, 2, 1)
#
# plt.imshow(mark_boundaries(image, segments))
#
# rag = rag_mean_color(image_, segments, mode="similarity")
#
# cut = ncut(segments, rag)
#
# print len(np.unique(cut))
#
# plt.subplot(1, 2, 2)
#
# plt.imshow(mark_boundaries(image, cut))
#
# plt.show()
#
# path = path.split('.')[0]
#
# if not os.path.exists(path):
#
#     os.mkdir(path)
#
# for i, superpixel in enumerate(np.unique(cut)):
#
#     plt.figure(i)
#
#     img = rgb2rgba(image)
#
#     img[cut == superpixel] = (0, 255, 0, 100)
#
#     plt.imshow(img)
#
#     plt.savefig("%s/%d.png" % (path, i))
