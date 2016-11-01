# -*- coding: utf-8 -*-

"""

    stamaimer 09/26/16

"""

import sys
import random
import logging

from os import mkdir, path
# from collections import Counter

from skimage import io
from skimage.color import label2rgb
from skimage.measure import label, ransac, regionprops, EllipseModel
# from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.segmentation import clear_border, find_boundaries, mark_boundaries

from sklearn import cluster as Cluster

from scipy import ndimage

# from PIL import Image, ImageEnhance

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pymongo import MongoClient

client = MongoClient()

db = client.TubuleDetection.eigenvectors

np.set_printoptions(threshold=np.nan)

logging.basicConfig(format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
                    stream=sys.stdout, level=logging.INFO)

IMAGE = None

PATH2IMAGE = None

W = H = D = 0

CLUSTER_NUM = 3


def show(image):

    """show image"""

    plt.subplot(111)

    plt.imshow(image)

    plt.show()


def show_with_rect(labeled, postfix):

    """show image with rect"""

    _, ax = plt.subplots(1, 1)

    ax.imshow(mark_boundaries(label2rgb(labeled, image=IMAGE), labeled, mode="inner"))

    for region in regionprops(labeled):

        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=0, edgecolor='red', linewidth=0.5)

        ax.add_patch(rect)

    plt.savefig(path.splitext(PATH2IMAGE)[0] + postfix + ".png")

    # plt.show()


def scatter(centroid, boundary, neighborhood):

    """scatter"""

    # plt.subplot(111)

    figure, ax = plt.subplots(1, 1)

    _image = IMAGE

    ax.imshow(_image)

    ax.scatter(centroid[1], centroid[0], c='g')

    boundary = boundary.T

    ax.scatter(boundary[1], boundary[0], s=0.1, marker='.')

    neighborhood = neighborhood.T

    ax.scatter(neighborhood[1], neighborhood[0])

    path2save = path.splitext(PATH2IMAGE)[0]

    if not path.exists(path2save):

        mkdir(path2save)

    figure.savefig(path2save + "/" + str(centroid) + ".png")

    plt.close(figure)

    # plt.show()


def rgb2gray(rgb):

    """rgb to gray"""

    return 0.2125 * rgb[0] + 0.7154 * rgb[1] + 0.0721 * rgb[2]


def preprocess(image):

    """preprocess"""

    image = image[:, :, 0:3]

    # image = equalize_hist(image)

    # image = equalize_adapthist(image)

    # p2, p98 = np.percentile(image, (2, 98))
    #
    # image = rescale_intensity(image, in_range=(p2, p98))

    # image = ImageEnhance.Sharpness(Image.fromarray(image))
    #
    # image = np.asarray(image.enhance(5))

    return image


def cluster(pixels):

    """cluster pixels"""

    kmeans = Cluster.KMeans(n_clusters=CLUSTER_NUM)

    kmeans.fit(pixels)  # random choice

    labels = kmeans.labels_.reshape((W, H))

    graies = [(rgb2gray(np.average(IMAGE[labels == i], axis=0)), i) for i in xrange(CLUSTER_NUM)]

    graies.sort()

    return labels, graies


def detect_lumens(close_radius, open_radius):

    """detect lumens"""

    lumens = np.zeros((W, H))

    lumens[labels == graies[2][1]] = 1

    cleaned = clear_border(lumens)

    closed = binary_closing(cleaned, disk(close_radius))

    filled = ndimage.binary_fill_holes(closed)

    opened = binary_opening(filled, disk(open_radius))

    labeled = label(opened)

    # show_with_rect(labeled, "L")

    lumens = dict()

    areas = np.array([region.area for region in regionprops(labeled)])

    # counter = Counter(areas)
    #
    # plt.subplot(111)
    #
    # plt.hist(areas, np.arange(areas.min(), areas.max(), 1000))
    #
    # plt.show()

    for region in regionprops(labeled):

        if region.area >= areas.mean():

            mask = np.zeros((W, H))

            mask[labeled == region.label] = 1

            boundary = np.transpose(np.nonzero(find_boundaries(mask, mode="inner")))

            lumens[region.centroid] = (boundary, region.area)

    return lumens


def detect_nucleis(open_radius, labels, graies):

    """detect nucleis"""

    image_ = np.copy(IMAGE)

    image_[labels == graies[1][1]] = 255

    image_[labels == graies[2][1]] = 255

    pixels = np.reshape(image_, (W * H, D))

    labels, graies = cluster(pixels)

    nucleis = np.zeros((W, H))

    nucleis[labels == graies[0][1]] = 1

    cleaned = clear_border(nucleis)

    filled = ndimage.binary_fill_holes(cleaned)

    opened = binary_opening(filled, disk(open_radius))

    labeled = label(opened)

    # show_with_rect(labeled, "N")

    nucleis = np.array([region.centroid for region in regionprops(labeled)])

    return nucleis


def construct_neighborhood(lumens, nucleis):

    """construct neighborhood"""

    records = dict()

    for lumen in lumens:

        neighborhood = list()

        for coordinate in lumens[lumen][0][::2]:

            for centroid in nucleis:

                if np.linalg.norm(coordinate - centroid) <= 50:

                    neighborhood.append(tuple(centroid))

                    break

        neighborhood = np.array(list(set(neighborhood)))

        if len(neighborhood):

            scatter(lumen, lumens[lumen][0], neighborhood)

            records[lumen] = (neighborhood, lumens[lumen][1])

    return records


def generate_eigenvectors(name, records):

    """generate eigenvectors"""

    # eigenvectors = list()

    for centroid in records:

        eigenvector = list()

        neighborhood = records[centroid][0]

        area = records[centroid][1]

        eigenvector.append(len(neighborhood) / float(area))

        distance = [np.linalg.norm(centroid - item) for item in neighborhood]

        eigenvector.extend([np.std(distance), np.mean(distance),
                            max(distance), (min(distance), max(distance))])

        eigenvector.append(np.mean([item - max(distance) for item in distance]))

        eigenvector.append(np.mean([item - min(distance) for item in distance]))

        eigenvector.append(np.mean([item - np.mean(distance) for item in distance]))

        eigenvector.append(np.mean([item - np.median(distance) for item in distance]))

        angles = list()

        for ids, i in enumerate(neighborhood):

            for j in neighborhood[ids+1:]:

                oi = i - centroid

                oj = j - centroid

                cosine = np.dot(oi, oj)/np.linalg.norm(oi)/np.linalg.norm(oj)

                angle = np.arccos(np.clip(cosine, -1.0, 1.0))

                angles.append(angle)

        eigenvector.extend([np.std(angles), np.mean(angles)])

        interval = [np.linalg.norm(i - j) \
                    for ids, i in enumerate(neighborhood) \
                    for j in neighborhood[ids+1:]]

        eigenvector.extend([np.std(interval), np.mean(interval)])

        # print len(neighborhood), neighborhood

        # ransac_model, inliers = ransac(neighborhood, EllipseModel, 5, 3, max_trials=50)
        #
        # eigenvector.extend([ransac_model.params[2], ransac_model.params[3]])

        # logging.info(eigenvector)

        logging.info(db.insert_one({"name": name,
                                    "label": " ",
                                    "coordinate": centroid,
                                    "eigenvector": eigenvector}).inserted_id)

    #     eigenvectors.append(eigenvector)
    #
    # return eigenvectors


if __name__ == "__main__":

    try:

        logging.info("tik")

        PATH2IMAGE = sys.argv[1]

        IMAGE = io.imread(PATH2IMAGE)

        IMAGE = preprocess(IMAGE)

        W, H, D = IMAGE.shape

        pixels = np.reshape(IMAGE, (W * H, D))

        labels, graies = cluster(pixels)

        logging.info(".")

        lumens = detect_lumens(int(sys.argv[2]), int(sys.argv[3]))

        logging.info(".")

        nucleis = detect_nucleis(int(sys.argv[4]), labels, graies)

        logging.info(".")

        records = construct_neighborhood(lumens, nucleis)

        logging.info(".")

        vectors = generate_eigenvectors(PATH2IMAGE, records)

        logging.info("tok")

    except IndexError:

        print "Usage: python palceholder.py PATH2IMAGE"
