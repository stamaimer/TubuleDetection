# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import argparse
import logging
import skimage
import random
import numpy
import math
import sys
import os

import b2ac.preprocess
import b2ac.fit
import b2ac.conversion

from PIL import Image

from sklearn.utils import shuffle
from sklearn.cluster import KMeans

from skimage import io, color, data, filters, feature, img_as_float
from skimage.color import rgb2gray, separate_stains  # , rgba2rgb
from skimage.future import graph
from skimage.filters import gaussian
from skimage.measure import moments, moments_central, ransac, EllipseModel
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

    for i in range(len(images)):

        plt.figure(i)

        plt.imshow(images[i])

    plt.show()


def rgb2rgba(img):

    img_pil_rgb = Image.fromarray(img)

    img_pil_rgba = img_pil_rgb.convert("RGBA")

    img_rgba = numpy.asarray(img_pil_rgba)

    img_rgba.setflags(write=1)

    return img_rgba


def detect_nuclear(path):

    nuclear_centroid = []

    image = io.imread(path)

    # image = rgba2rgb(image)

    # color deconvolution & adjust intensity levels

    hematoxylin = separate_stains(image, conv_matrix)[:, :, 0]

    rescaled = rescale_intensity(hematoxylin)  # ?

    # morphology opening operation

    selem = disk(1)  # adjust the type & size of selem

    opened = opening(rescaled, selem)

    # blobs detection

    blobs_doh = feature.blob_doh(opened, max_sigma=30, threshold=.01)  #

    for blob in blobs_doh:

        y, x, _ = blob

        nuclear_centroid.append((x, y))

    # prepare to show images
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

    records = dict()

    image = io.imread(path)

    # image = rgba2rgb(image)

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

        cr = moments_[0, 1] / moments_[0, 0]  # y

        cc = moments_[1, 0] / moments_[0, 0]  # x

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

        if len(neighborhood):

            plt.figure(i)

            img = rgb2rgba(image)

            img[cut == superpixel] = (0, 255, 0, 100)

            plt.imshow(img)

            # plt.imshow(opened)

            plt.scatter(numpy.array(neighborhood[1]), numpy.array(neighborhood[0]), s=3, c='y')

            plt.xlim(0, 1360)

            plt.ylim(1024, 0)

            path = path.split('.')[0]

            if not os.path.exists(path):

                os.mkdir(path)

            plt.savefig("%s/%d.png" % (path, i))

            logging.info("%s/%d.png" % (path, i))

            # plt.show()

        boundaries.append(boundary)

        if len(neighborhood):

            records[(cc, cr)] = numpy.transpose(neighborhood)

    return records  # , boundaries


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


def generate_eigenvectors(records):

    eigenvectors = []

    for lumen_centroid in records:

        eigenvector = []

        neighborhoods = records[lumen_centroid]

        eigenvector.append(len(neighborhoods))  # 1

        distance = [ numpy.linalg.norm(lumen_centroid - neighborhood) for neighborhood in neighborhoods ]

        eigenvector.extend([numpy.mean(distance), numpy.std(distance), random.shuffle(distance), max(distance), (min(distance), max(distance))])  # 2-6

        eigenvector.append(numpy.mean([ item - max(distance) for item in distance ]))  # 7

        eigenvector.append(numpy.mean([ item - min(distance) for item in distance ]))  # 8

        eigenvector.append(numpy.mean([ item - numpy.mean(distance) for item in distance ]))  # 9

        eigenvector.append(numpy.mean([ item - numpy.median(distance) for item in distance ]))  # 10

        angles = []

        for i, neighborhood_i in enumerate(neighborhoods):

            for neighborhood_j in neighborhoods[i+1:]:

                cosine = numpy.dot(neighborhood_i, neighborhood_j)/numpy.linalg.norm(neighborhood_i)/numpy.linalg.norm(neighborhood_j)

                angle = numpy.arccos(numpy.clip(cosine, -1.0, 1.0))

                angles.append(angle)

        eigenvector.extend([numpy.mean(angles), numpy.std(angles), random.shuffle(angles)])  # 11-13

        distance_ = [ numpy.linalg.norm(neighborhood_i - neighborhood_j) for i, neighborhood_i in enumerate(neighborhoods) for neighborhood_j in neighborhoods[i+1:] ]

        eigenvector.extend([numpy.mean(distance_), numpy.std(distance_), random.shuffle(distance_)])  # 14-16

        print neighborhoods

        ransac_model, inliers = ransac(neighborhoods, EllipseModel, 5, 3, max_trials=50)

        print ransac_model.params

        # _, x_mean, y_mean = b2ac.preprocess.remove_mean_values(neighborhoods)
        #
        # print neighborhoods, x_mean, y_mean
        #
        # conic_numpy = b2ac.fit.fit_improved_B2AC_numpy(neighborhoods)
        #
        # general_form_numpy = b2ac.conversion.conic_to_general_1(conic_numpy)
        # general_form_numpy[0][0] += x_mean
        # general_form_numpy[0][1] += y_mean
        #
        # print general_form_numpy

        # conic_double = b2ac.fit.fit_improved_B2AC_double(neighborhoods)
        #
        # general_form_double = b2ac.conversion.conic_to_general_1(conic_double)
        # general_form_double[0][0] += x_mean
        # general_form_double[0][1] += y_mean
        #
        # print general_form_double

        eigenvectors.append(eigenvector)


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("path", help="")

    args = argument_parser.parse_args()

    path = args.path

    logging.info("tik")

    logging.info(skimage.__version__)

    nuclears_centroids = detect_nuclear(path)

    lumenes_boundaries = detect_lumenes(path, nuclears_centroids)  # just for tidy

    # neighborhoods = ConstructNeighborhood(nuclear_centroids, lumen_boundaries)

    # generate_eigenvectors(lumenes_boundaries)

    logging.info("tok")
