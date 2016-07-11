# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import argparse
import logging
import skimage
import numpy
import math
import sys

from sklearn.utils import shuffle
from sklearn.cluster import KMeans

from skimage import io, color, data, filters, feature, img_as_float
from skimage.color import separate_stains
from skimage.future import graph
from skimage.filters import gaussian
from skimage.measure import moments, moments_central
from skimage.exposure import rescale_intensity
from skimage.morphology import disk, diamond, star, binary_opening, dilation, erosion, opening
from skimage.segmentation import active_contour, find_boundaries, mark_boundaries, quickshift

numpy.set_printoptions(threshold=numpy.nan)

conv_matrix = numpy.array([[0.644211, 0.716556, 0.266844],
                           [0.092789, 0.954111, 0.283111],
                           [0.000000, 0.000000, 0.000000]])

logging.basicConfig(format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
                    stream=sys.stdout, level=logging.INFO)


def show(images):

    for i in xrange(len(images)):

        plt.figure(i)

        plt.imshow(images[i])

    plt.show()


def detect_nuclear(path):

    nuclear_centroid = []

    image = io.imread(path)

    # color deconvolution & adjust intensity levels

    hematoxylin = separate_stains(image, conv_matrix)[:, :, 0]

    rescaled = rescale_intensity(hematoxylin)

    # morphology opening operation

    selem = disk(1)  # adjust the type & size of selem

    opened = opening(rescaled, selem)

    # blobs detection

    blobs_doh = feature.blob_doh(opened, max_sigma=30, threshold=.01)  #

    for blob in blobs_doh:

        y, x, _ = blob

        nuclear_centroid.append((x, y))

    # # prepare to show images
    #
    # images = [image, rescaled, opened, opened]
    #
    # titles = ["Origin Image", "Color Deconvolution", "Nuclear Detection"]
    #
    # _, axes = plt.subplots(1, 3)
    #
    # plt.tight_layout()
    #
    # axes = axes.ravel()
    #
    # sequence = zip(images, titles, axes)
    #
    # for image, title, ax in sequence:
    #
    #     ax.set_title(title)
    #
    #     ax.imshow(image, interpolation="nearest")
    #
    #     ax.set_axis_off()
    #
    # # mark boundaries
    #
    # for blob in blobs_doh:
    #
    #     y, x, r = blob
    #
    #     c = plt.Circle((x, y), r, color="red", linewidth=0.5, fill=False)
    #
    #     axes[-1].add_patch(c)
    #
    # plt.show()

    logging.info("The number of nuclears is %d." % len(nuclear_centroid))

    return nuclear_centroid


def illumination_invariant_representation(path):

    image = io.imread(path)

    image = img_as_float(image)

    for x in xrange(image.shape[0]):

        for y in xrange(image.shape[1]):

            rgb = image[x][y]

            tmp = map(lambda channel: channel-min(rgb), rgb)

            foo = sum(map(lambda channel: math.pow(channel, 2), tmp))

            if foo == 0:

                continue

            else:

                image[x][y] = map(lambda channel: channel/float(math.sqrt(foo)), tmp)

    # show([image])

    return image


def recreate_image(codebook, labels, w, h):

    d = codebook.shape[1]

    image = numpy.zeros((w, h, d))

    label_idx = 0

    for i in xrange(w):

        for j in xrange(h):

            image[i][j] = codebook[labels[label_idx]]

            label_idx += 1

    return image


def quantization(path):

    cluster_number = 3

    image = io.imread(path)

    image = img_as_float(image)

    w, h, d = tuple(image.shape)

    samples_number = w * h / 3

    image_array = numpy.reshape(image, (w * h, d))

    image_array_sample = shuffle(image_array, random_state=0)[:samples_number]

    kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(image_array_sample)

    lables = kmeans.predict(image_array)

    image = recreate_image(kmeans.cluster_centers_, lables, w, h)

    # show([image])

    return image


def detect_lumenes(path, nuclear_centroids):

    boundaries = []

    image = io.imread(path)

    logging.info(image.shape)

    # image = data.coffee()

    clustered = quickshift(image, kernel_size=10, max_dist=30, sigma=0)  # sigma # automatic tuning

    logging.info("The number of clusters is %d." % len(numpy.unique(clustered)))

    rag = graph.rag_mean_color(image, clustered, mode="similarity")  # sigma # connectivity

    cut = graph.cut_normalized(clustered, rag)  # thresh # num_cuts # max_edge

    logging.info("The number of clusters is %d." % len(numpy.unique(cut)))

    # boundaries = numpy.transpose(numpy.nonzero(find_boundaries(cut, mode="inner")))  # coordinate of boundaries
    #
    # snake = active_contour(gaussian(image, sigma=1), boundaries.astype(float))  # coordinate of boundaries

    # show([mark_boundaries(image, cut, mode="inner")])

    for (i, superpixel) in enumerate(numpy.unique(cut)):

        mask = numpy.zeros(image.shape[:2])

        mask[cut == superpixel] = 1

        # show([mask])

        moments_ = moments(mask)

        # cr = moments_[0, 1] / moments_[0, 0]  # y
        #
        # cc = moments_[1, 0] / moments_[0, 0]  # x

        if moments_[0, 0] > 100000:  # area

            continue

        # radius = 0.5 * numpy.sqrt(4 * moments_[0, 0] / numpy.pi)

        selem = disk(3)  # adjust the type & size of selem

        opened = binary_opening(mask, selem)

        # clustered = quickshift(mask, kernel_size=10, max_dist=30, sigma=0)
        #
        # rag = graph.rag_mean_color(mask, clustered, mode="similarity")
        #
        # cut = graph.cut_normalized(clustered, rag)
        #
        # show([mark_boundaries(mask, cut, mode="inner")])
        #
        # boundaries = find_boundaries(opened, mode="inner")
        #
        # show([mark_boundaries(opened, boundaries)])

        boundary = numpy.transpose(numpy.nonzero(find_boundaries(opened, mode="inner")))  # coordinate of boundaries

        neighborhood = []

        for lumen_boundary_coordinate in boundary:

            for nuclear_centroid_coordinate in nuclear_centroids:

                if numpy.linalg.norm(lumen_boundary_coordinate - nuclear_centroid_coordinate) <= 15:

                    neighborhood.append(nuclear_centroid_coordinate)

        neighborhood = numpy.transpose(list(set(neighborhood)))

        print neighborhood

        if len(neighborhood):

            show([opened])

            plt.scatter(numpy.array(neighborhood[0]), numpy.array(neighborhood[1]))

            plt.xlim(0, 1360)

            plt.ylim(1024, 0)

            plt.show()

        boundaries.append(boundary)

    return boundaries


def ConstructNeighborhood(nuclear_centroids, lumen_boundaries):

    neighborhoods = []

    for lumen_boundary in lumen_boundaries:

        neighborhood = []

        for lumen_boundary_coordinate in lumen_boundary:

            for nuclear_centroid_cordinate in nuclear_centroids:

                if numpy.linalg.norm(lumen_boundary_coordinate - nuclear_centroid_cordinate) <= 1:

                    neighborhood.append(nuclear_centroid_cordinate)

        neighborhoods.append(neighborhood)

    return neighborhoods


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("path", help="")

    args = argument_parser.parse_args()

    logging.debug(skimage.__version__)

    path = args.path

    logging.info("tik")

    nuclear_centroids = detect_nuclear(path)

    lumen_boundaries = detect_lumenes(path, nuclear_centroids)

    # neighborhoods = ConstructNeighborhood(nuclear_centroids, lumen_boundaries)

    logging.info("tok")
