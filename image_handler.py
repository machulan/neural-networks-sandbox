from PIL import Image

import neural_network

import numpy as np
import math
import metrics


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
    handled_image_data = neural_network.run(image_data)

    return get_image(handled_image_data, mode='L')

    # TODO shelve or pickle для запоминания нейронной сети IT DOESNT WORK
    # TODO h5py for saving model




    # stub = Image.open('../neural-networks-sandbox/images/doctorwho.jpg')

    stub = image.resize((image.width * 2, image.height * 2))

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
    print('converting to rgb running...')
    return image.convert('RGB')


def get_images_difference(true_image, pred_image, metric='psnr'):
    if metric == 'psnr':
        true_image_data = get_image_data(true_image)
        pred_image_data = get_image_data(pred_image)
        return metrics.psnr(true_image_data, pred_image_data)
    else:
        print('Unknown metric ' + str(metric))
        exit()


def get_image_data(image):
    image_data = list(image.getdata())
    image_data = np.array(image_data)
    print('image_data shape before changing:', image_data.shape)
    if image_data.ndim == 1:
        print('Getting L image data...')
        image_data.shape = (image.height, image.width)
        # exit()
    else:
        print('Getting RGB image data...')
        image_data.shape = (image.height, image.width, 3)
    print('image_data shape after changing:', image_data.shape)
    return image_data.tolist()


def get_image_head_data(image):
    return list(image.getdata())[:10]


def zoom_out_image(image, times=2.0):
    print('zooming out the image')
    return image.resize((int(image.width / times), int(image.height / times)))


def show_mnist_example():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    for i in range(1):
        get_image(X_train[i].tolist())


def show_cifar10_example():
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # num_train, height, width, depth = X_train.shape
    print(X_train.shape)
    # print(X_train[0].tolist())

    for i in range(5):
        get_image(X_train[i].tolist(), mode='RGB')


def handle_mnist():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train[0].tolist()

    # image_data = X_test[0].tolist()
    # image = get_image(image_data, mode='L')
    # image.show()
    # return

    dataset_size = 60000

    dataset_X, dataset_Y = [], []
    for i, X_train_item in enumerate(X_train[:dataset_size]):
        image_data = X_train_item.tolist()
        image = get_image(image_data, mode='L')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_out_image_data = get_image_data(zoomed_out_image)

        dataset_X.append(zoomed_out_image_data)
        dataset_Y.append(image_data)

        print('image ' + str(i) + ', height : ' + str(len(image_data)), 'width : ' + str(len(image_data[0])), sep=', ')
        print('zoomed_out_image ' + str(i) + ', height : ' + str(len(zoomed_out_image_data)),
              'width : ' + str(len(zoomed_out_image_data[0])), sep=', ')
        # print('image_data :', image_data)
        # print('zoomed_out_image_data :', zoomed_out_image_data)
        # image.show()
        # zoomed_out_image.show()

    dataset = (dataset_X, dataset_Y)

    import pickle
    mnist_dataset_file = open('datasets/mnist-dataset.pkl', 'wb')
    pickle.dump(dataset, mnist_dataset_file)
    mnist_dataset_file.close()


    # zoomed_out_image.show()


if __name__ == '__main__':
    print('image_handler module running...')

    # handle_mnist()
    from neural_network import get_mnist_dataset
    # dataset_X, dataset_Y = get_mnist_dataset()
    # print('dataset_X', len(dataset_X))
    # print('dataset_Y', len(dataset_Y))
    # (X_train, Y_train), (X_test, Y_test) = get_mnist_dataset()
    # print('X_train', len(X_train))
    # print('Y_train', len(Y_train))
    # print('X_test', len(X_test))
    # print('Y_test', len(Y_test))

    path_to_images = 'images/'
    get_images_difference()


    # show_mnist_example()
    # show_cifar10_example()


    # image_data = [[255, 255, 255], [0, 0, 0], [100, 100, 100], [100, 100, 100]]
    # image_data = np.array(image_data).astype('uint8')
    # print(image_data)
    # # image_data = np.reshape(image_data, 256, 256)
    # image = Image.fromarray(image_data, mode='L')
    # image.show()

    # preparing the image dataset
