from PIL import Image

import neural_network

import numpy as np
import math
import metrics

from skimage.measure import compare_psnr, compare_ssim, compare_mse


def print_image_data1(image):
    print('Image data:')
    print('format:', image.format)
    print('mode:', image.mode)
    print('width:', image.width)
    print('height:', image.height)


def print_image_data(image):
    print('IMAGE DATA:')
    print('format:', image.format, 'mode:', image.mode)
    print('width:', image.width, 'height:', image.height)


def handle_image(image):
    print('image handling running...')

    print_image_data(image)

    image_data = get_image_data(image)
    # handled_image_data = neural_network.run(image_data)

    # return get_image(handled_image_data, mode='RGB')
    # return get_image(handled_image_data, mode='L')

    # TODO shelve or pickle для запоминания нейронной сети IT DOESNT WORK
    # TODO h5py for saving model




    # stub = Image.open('../neural-networks-sandbox/images/doctorwho.jpg')

    stub = image.resize((image.width * 2, image.height * 2))
    return stub

    print(get_image_head_data(stub))
    # print(list(map(lambda p: (p[0] + p[1] + p[2]) // 3, get_image_head_data(stub))))
    stub = convert_to_white_black(image)
    print(get_image_head_data(stub))

    # stub = convert_to_rgb(stub)
    # print(list(stub.getdata())[:10])

    return stub


def get_image(image_data, mode='L'):
    image_data = np.array(image_data).astype('uint8')
    image = Image.fromarray(image_data, mode=mode)
    # image.show()
    return image


    # img.resize((x, y))
    #  img.putpixel((1, 0), (0, 0, 0))
    # print(img.getpixel((1, 0)))


def convert_to_white_black(image):
    print('converting to white-black running...')
    return image.convert('L')


def convert_to_rgb(image):
    # print('converting to rgb running...')
    return image.convert('RGB')


def convert_to_YCbCr(image):
    print('converting to YCbCr running...')
    return image.convert('YCbCr')


def convert_l_image_data_to_rgb_image_data(l_image_data):
    image = get_image(l_image_data, mode='L')
    rgb_image = convert_to_rgb(image)
    return get_image_data(rgb_image)


def get_images_difference(true_image, pred_image, metric='psnr'):
    if metric == 'psnr':
        true_image_data = get_image_data(true_image)
        pred_image_data = get_image_data(pred_image)
        return metrics.psnr(true_image_data, pred_image_data)
    else:
        print('Unknown metric ' + str(metric))
        exit()


def get_images_difference_metrics(true_image, pred_image):
    print('computing images difference metrics...')
    true_image = convert_to_rgb(true_image)
    test_image = convert_to_rgb(pred_image)
    true_image_data = get_image_data(true_image)
    test_image_data = get_image_data(test_image)
    true_image_data = np.array(true_image_data, dtype='uint8')
    test_image_data = np.array(test_image_data, dtype='uint8')

    # print(true_image_data.shape)
    # print(test_image_data.shape)

    psnr = compare_psnr(true_image_data, test_image_data)
    ssim = compare_ssim(true_image_data, test_image_data, multichannel=True)
    mse = compare_mse(true_image_data, test_image_data)

    return psnr.item(), ssim.item(), mse.item()


def get_image_data(image, mode='L'):
    image_data = list(image.getdata())
    image_data = np.array(image_data)
    # print('image_data shape before changing:', image_data.shape)
    if image_data.ndim == 1:
        # print('getting L image data...')
        image_data.shape = (image.height, image.width)
        # exit()
    elif image_data.ndim == 2:
        # print('getting RGB image data...')
        image_data.shape = (image.height, image.width, 3)
    # print('image_data shape after changing:', image_data.shape)
    return image_data.tolist()


def get_image_head_data(image):
    return list(image.getdata())[:10]


def zoom_out_image(image, times=2.0):
    # print('zooming out the image')
    return image.resize((int(image.width / times), int(image.height / times)))


def zoom_up_image(image, times=2.0):
    # print('zooming up the image')
    return image.resize((int(image.width * times), int(image.height * times)))


if __name__ == '__main__':
    print('image_handler module running...')

    # image_data = [[255,123,10,0]]
    # image = get_image(image_data, mode='L')
    # # image.show()
    #
    # image_data = [[0,0,0, 0], [100,100,100,100], [255,255,255]]
    image_data = [[[0,50,0], [255,255,0], [234,0,234]]]
    image = get_image(image_data, mode='RGB')
    image.show()


    # handle_mnist()
    # from neural_network import get_mnist_dataset

    # dataset_X, dataset_Y = get_mnist_dataset()
    # print('dataset_X', len(dataset_X))
    # print('dataset_Y', len(dataset_Y))
    # (X_train, Y_train), (X_test, Y_test) = get_mnist_dataset()
    # print('X_train', len(X_train))
    # print('Y_train', len(Y_train))
    # print('X_test', len(X_test))
    # # print('Y_test', len(Y_test))
    #
    # path_to_images = 'images/'
    # get_images_difference()

    # make_srcnn_dataset_based_on_mnist()


    # show_mnist_example()
    # show_cifar10_example()


    # image_data = [[255, 255, 255], [0, 0, 0], [100, 100, 100], [100, 100, 100]]
    # image_data = np.array(image_data).astype('uint8')
    # print(image_data)
    # # image_data = np.reshape(image_data, 256, 256)
    # image = Image.fromarray(image_data, mode='L')
    # image.show()

    # preparing the image dataset
