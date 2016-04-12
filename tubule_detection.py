# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import argparse
import numpy
import math

from skimage import io, filters, feature
from skimage.color import separate_stains
from skimage.exposure import rescale_intensity
from skimage.morphology import disk, diamond, square, star, opening 
from skimage.segmentation import mark_boundaries, felzenszwalb, quickshift, slic

numpy.set_printoptions(threshold=numpy.nan)

conv_matrix = numpy.array([[0.644211, 0.716556, 0.266844],
                           [0.092789, 0.954111, 0.283111],
                           [0.000000, 0.000000, 0.000000]])


def show(images):

    for i in xrange(len(images)):

        plt.figure(i)

        plt.imshow(images[i])

    plt.show()


def detect_nuclear(path):

    image = io.imread(path)

    # color deconvolution & adjust intensity levels

    hematoxylin = separate_stains(image, conv_matrix)[:, :, 0]

    rescaled = rescale_intensity(hematoxylin)

    # morphology opening operation

    selem = disk(1)  # adjust the type & size of selem

    opened = opening(rescaled, selem)

    # blobs detection
    #
    # blobs_log = feature.blob_log(opened)
    # 
    # blobs_log[:, 2] = blobs_log[:, 2] * math.sqrt(2)
    # 
    # blobs_dog = feature.blob_dog(opened)
    # 
    # blobs_dog[:, 2] = blobs_dog[:, 2] * math.sqrt(2)

    blobs_doh = feature.blob_doh(opened, max_sigma=30, threshold=.01)  # 

    # prepare to show images

    images = [image, rescaled, opened, opened]

    titles = ["Origin Image", "Color Deconvolution", "Morphology Opened", "Nuclear Detection"]

    _, axes = plt.subplots(2, 2)

    plt.tight_layout()

    axes = axes.ravel()

    sequence = zip(images, titles, axes)

    for image, title, ax in sequence:

        ax.set_title(title)

        ax.imshow(image, interpolation="nearest")

        ax.set_axis_off()

    # mark boundaries

    for blob in blobs_doh:

        y, x, r = blob

        c = plt.Circle((x, y), r, color="red", linewidth=0.5, fill=False)

        axes[-1].add_patch(c)

    plt.show()


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("path", help="")

    args = argument_parser.parse_args()

    path = args.path

    detect_nuclear(path)
