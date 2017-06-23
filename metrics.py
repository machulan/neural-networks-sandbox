import math
import numpy as np

"""Metrics for evaluating image difference"""


def mse(true_image_data, pred_image_data):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    if true_image_data.shape != pred_image_data.shape:
        print('Trying to compare images with different dimensions.')
        exit()

    result = np.sum((true_image_data.astype('float32') - pred_image_data.astype('float32')) ** 2)
    result /= float(true_image_data.size)

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return result


def psnr(true_image_data, pred_image_data):
    return 20 * math.log10(np.max(true_image_data) / math.sqrt(mse(true_image_data, pred_image_data)))

