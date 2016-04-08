# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import argparse
import numpy
import math

from skimage import io, filters
from skimage.color import separate_stains, rgb2gray, rgb2grey
from skimage.exposure import rescale_intensity
from skimage.morphology import opening, disk

HE1_conv_matrix = numpy.array([[0.644211, 0.716556, 0.266844],
                               [0.092789, 0.954111, 0.283111],
                               [0.000000, 0.000000, 0.000000]])

HE2_conv_matrix = numpy.array([[0.49015734, 0.76897085, 0.41040173],
                               [0.04615336, 0.84206840, 0.53739250],
                               [0.00000000, 0.00000000, 0.00000000]])


cos = numpy.zeros((3, 3))

len = numpy.zeros(3)

q = numpy.zeros(9)


def color_deconvolution(path, conv_matrix):

    # for c in xrange(3):
    #
    #     len[c] = math.sqrt(reduce(lambda x, y: x + y, map(lambda z: z*z, conv_matrix[:, c])))
    #
    #     if len[c] != 0.0:
    #
    #         for r in xrange(3):
    #
    #             cos[r, c] = conv_matrix[r, c] / len[c]
    #
    # if sum(cos[:, 1]) == 0.0:
    #
    #     cos[0, 1] = cos[2, 0]
    #     cos[1, 1] = cos[0, 0]
    #     cos[2, 1] = cos[1, 0]
    #
    # if sum(cos[:, 2]) == 0.0:
    #
    #     for r in xrange(3):
    #
    #         temp = reduce(lambda x, y: x + y, map(lambda z: z*z, cos[r, :]))
    #
    #         if temp < 1:
    #
    #             cos[r, 2] = math.sqrt(1.0 - temp)
    #
    # leng = math.sqrt(reduce(lambda x, y: x + y, map(lambda z: z*z, cos[:, 2])))
    #
    # for r in xrange(3):
    #
    #     cos[r, 2] /= leng
    #
    # for r in xrange(3):
    #
    #     cos[r] = map(lambda x: 0.001 if x == 0.0 else x, cos[r, :])
    #
    # A = cos[1, 1] - cos[0, 1] * cos[1, 0] / cos[0, 0]
    # V = cos[2, 1] - cos[0, 1] * cos[2, 0] / cos[0, 0]
    # C = cos[2, 2] - cos[1, 2] * V / A + cos[0, 2] * (V / A * cos[1, 0] / cos[0, 0] - cos[2, 0] / cos[0, 0])
    #
    # q[2] = (-cos[0, 2] / cos[0, 0] - cos[0, 2] / A * cos[0, 1] / cos[0, 0] * cos[1, 0] / cos[0, 0] + cos[1, 2] / A * cos[0, 1] / cos[0, 0]) / C
    # q[1] = -q[2] * V / A - cos[0, 1] / (cos[0, 0] * A)
    # q[0] = 1.0 / cos[0, 0] - q[1] * cos[1, 0] / cos[0, 0] - q[2] * cos[2, 0] / cos[0, 0]
    # q[5] = (-cos[1, 2] / A + cos[0, 2] / A * cos[1, 0] / cos[0, 0]) / C
    # q[4] = -q[5] * V / A + 1.0 / A
    # q[3] = -q[4] * cos[1, 0] / cos[0, 0] - q[5] * cos[2, 0] / cos[0, 0]
    # q[8] = 1.0 / C
    # q[7] = -q[8] * V / A
    # q[6] = -q[7] * cos[1, 0] / cos[0, 0] - q[8] * cos[2, 0] / cos[0, 0]

    # print numpy.linalg.inv(conv_matrix)

    image = io.imread(path)

    hematoxylin = separate_stains(image, conv_matrix)[:, :, 0]

    rescaled = rescale_intensity(hematoxylin, out_range=(0, 1))

    edges = filters.sobel(rescaled)

    io.imshow(edges)

    io.show()

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="")

    argument_parser.add_argument("-p", "--path", help="")

    args = argument_parser.parse_args()

    path = args.path

    color_deconvolution(path, HE1_conv_matrix)
