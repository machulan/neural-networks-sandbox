import numpy as np
from image_handler import get_image, get_image_data, zoom_out_image, zoom_up_image
from neural_network import get_srcnn_mnist_dataset
from keras.datasets import mnist


def make_srcnn_dataset_based_on_mnist():
    print('making SRCNN dataset using mnist...')

    (Y_train, _), (Y_test, _) = mnist.load_data()  # 60000, 10000

    # making X_train list
    print('making X_train list...')
    X_train = []
    for item in Y_train:
        image_data = item.tolist()
        image = get_image(image_data, mode='L')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_train.append(zoomed_up_image_data)

    # making X_test list
    print('making X_test list...')
    X_test = []
    for item in Y_test:
        image_data = item.tolist()
        image = get_image(image_data, mode='L')
        zoomed_out_image = zoom_out_image(image, times=2)
        zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
        zoomed_up_image_data = get_image_data(zoomed_up_image)
        X_test.append(zoomed_up_image_data)

    # X_train, X_test = [], []
    # for X, Y in [(X_train, Y_train), (X_test, Y_test)]:
    #     for item in Y:
    #         image_data = item.tolist()
    #         image = get_image(image_data, mode='L')
    #         zoomed_out_image = zoom_out_image(image, times=2)
    #         zoomed_up_image = zoom_up_image(zoomed_out_image, times=2)
    #         zoomed_up_image_data = get_image_data(zoomed_up_image)
    #         X.append(zoomed_up_image_data)

    dtype = 'uint8'
    dataset = (np.array(X_train, dtype=dtype), np.array(Y_train, dtype=dtype)), \
              (np.array(X_test, dtype=dtype), np.array(Y_test, dtype=dtype))
    print('saving dataset to srcnn-mnist-dataset.npz...')
    np.savez('datasets/srcnn-mnist-dataset.npz', dataset)


def show_srcnn_mnist_dataset_example(count=1):
    (X_train, Y_train), (X_test, Y_test) = dataset = get_srcnn_mnist_dataset()

    for i in range(count):
        get_image(X_train[i], mode='L').show()
        get_image(Y_train[i], mode='L').show()

    for i in range(count):
        get_image(X_test[i], mode='L').show()
        get_image(Y_test[i], mode='L').show()


if __name__ == '__main__':
    print('dataset module running...')
    # make_srcnn_dataset_based_on_mnist()
    show_srcnn_mnist_dataset_example()
