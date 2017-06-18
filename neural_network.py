import shelve
import pickle

import numpy as np


def get_mnist_dataset():
    mnist_dataset_file = open('datasets/mnist-dataset.pkl', 'rb')
    dataset_X, dataset_Y = pickle.load(mnist_dataset_file)
    mnist_dataset_file.close()

    dataset_size = len(dataset_X) // 1
    dataset_X, dataset_Y = dataset_X[:dataset_size], dataset_Y[:dataset_size]

    train_size = int(len(dataset_X) * 0.9)
    X_train, X_test = dataset_X[:train_size], dataset_X[train_size:]
    Y_train, Y_test = dataset_Y[:train_size], dataset_Y[train_size:]

    return (np.array(X_train, dtype='uint8'), np.array(Y_train, dtype='uint8')), \
           (np.array(X_test, dtype='uint8'), np.array(Y_test, dtype='uint8'))


def print_result(train_result, test_result):
    print()
    print('test_result :', test_result)
    print('train_result :', 'epochs :', train_result.epoch, 'history :', train_result.history)


def fit_dense():
    print('neural network fitting...')

    from keras.layers import Input, Dense, Conv2D, MaxPooling2D
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l2  # L2-regularisation

    batch_size = 128  # in each iteration we consider 128 training examples at once
    num_epochs = 20  # we iterate twenty times over the entire training set
    hidden_size = 512
    kernel_size = 3  # we will use 3x3 kernels throughout
    conv_depth = 32  # use 32 kernels in both convolutional layers

    (X_train, Y_train), (X_test, Y_test) = get_mnist_dataset()

    num_X_train, height_X_train, width_X_train = X_train.shape
    num_X_test, height_X_test, width_X_test = X_test.shape

    print('num_X_train :', num_X_train, 'height_X_train :', height_X_train, 'width_X_train :', width_X_train,
          'num_X_test :', num_X_test)

    X_train = X_train.reshape(num_X_train, height_X_train * width_X_train)
    X_test = X_test.reshape(num_X_test, height_X_train * width_X_train)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255  # np.max(X_train)  # Normalise data to [0, 1] range
    X_test /= 255  # np.max(X_test)  # Normalise data ti [0, 1] range

    num_Y_train, height_Y_train, width_Y_train = Y_train.shape
    num_Y_test, height_Y_test, width_Y_test = Y_test.shape

    print('num_Y_train :', num_Y_train, 'height_Y_train :', height_Y_train, 'width_Y_train :', width_Y_train,
          'num_Y_test :', num_Y_test)

    Y_train = Y_train.reshape(num_Y_train, height_Y_train * width_Y_train)
    Y_test = Y_test.reshape(num_Y_test, height_Y_test * width_Y_test)
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    Y_train /= 255  # np.max(Y_train)  # Normalise data to [0, 1] range
    Y_test /= 255  # np.max(Y_test)  # Normalise data to [0, 1] range

    inp = Input(shape=(height_X_train * width_X_train,))
    hidden1 = Dense(hidden_size, activation='relu')(inp)
    hidden2 = Dense(hidden_size, activation='relu')(hidden1)
    hidden3 = Dense(hidden_size, activation='relu')(hidden2)
    hidden4 = Dense(hidden_size, activation='relu')(hidden3)
    hidden5 = Dense(hidden_size, activation='relu')(hidden4)
    hidden6 = Dense(hidden_size, activation='relu')(hidden5)
    hidden7 = Dense(hidden_size, activation='relu')(hidden6)
    hidden8 = Dense(hidden_size, activation='relu')(hidden7)
    out = Dense(height_Y_train * width_Y_train, activation='relu')(hidden8)

    model = Model(inputs=inp, outputs=out)

    # https://keras.io/losses/
    # https://keras.io/optimizers/

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    train_result = model.fit(X_train, Y_train,  # Train the model using the training set...
                             batch_size=batch_size,  # nb_epoch=num_epochs, verbose=0,
                             epochs=num_epochs,
                             verbose=1,
                             validation_split=0.0)  # ...holding out 10% of the data for validation

    test_result = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set

    print_result(train_result, test_result)

    prediction = model.predict(X_test, batch_size=batch_size, verbose=1)

    print()
    # PREDICTION IMAGE
    print('prediction image :')

    # prediction[0].resize((height_Y_test, width_Y_test))
    prediction.resize((num_Y_test, height_Y_test, width_Y_test))
    prediction = np.rint(prediction * 255).astype('uint8')
    print(prediction[0])
    print_ndarray_info(prediction)  # reshape((3, 4)) => a ; resize((2,6)) => on place
    print_ndarray_info(prediction[0])

    from keras.backend import clear_session
    clear_session()

    from image_handler import get_image
    prediction_image = get_image(prediction[0], mode='L')
    prediction_image.show(title='Prediction 28')

    # ZOOMED OUT IMAGE
    print('zoomed out image :')

    X_test.resize((num_X_test, height_X_test, width_X_test))
    X_test = np.rint(X_test * 255).astype('uint8')
    print_ndarray_info(X_test)
    print_ndarray_info(X_test[0])

    zoomed_out_image = get_image(X_test[0], mode='L')
    # zoomed_out_image.show(title='Zoomed out 14')

    # ORIGINAL IMAGE
    print('original image :')

    Y_test.resize((num_Y_test, height_Y_test, width_Y_test))
    Y_test = np.rint(Y_test * 255).astype('uint8')
    print_ndarray_info(Y_test)
    print_ndarray_info(Y_test[0])

    original_image = get_image(Y_test[0], mode='L')
    original_image.show(title='ORIGINAL IMAGE 28')







    # neural_network = shelve.open('neural_network')
    # # neural_network.clear()
    # model = neural_network.get('model', None)
    # if model is None:
    #     print('model is None')
    #     model = 'MODEL'
    #     neural_network['model'] = model
    # else:
    #     print(neural_network['model'])

def fit_conv():
    print('neural network fitting...')

    from keras.layers import Input, Dense, Conv2D, MaxPooling2D
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l2  # L2-regularisation

    batch_size = 128  # in each iteration we consider 128 training examples at once
    num_epochs = 20 # we iterate twenty times over the entire training set
    hidden_size = 512
    kernel_size = 3  # we will use 3x3 kernels throughout
    conv_depth = 32  # use 32 kernels in both convolutional layers
    depth = 1

    (X_train, Y_train), (X_test, Y_test) = get_mnist_dataset()

    num_X_train, height_X_train, width_X_train = X_train.shape
    num_X_test, height_X_test, width_X_test = X_test.shape

    print('num_X_train :', num_X_train, 'height_X_train :', height_X_train, 'width_X_train :', width_X_train,
          'num_X_test :', num_X_test)

    X_train = X_train.reshape(num_X_train, height_X_train * width_X_train)
    X_test = X_test.reshape(num_X_test, height_X_train * width_X_train)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255  # np.max(X_train)  # Normalise data to [0, 1] range
    X_test /= 255  # np.max(X_test)  # Normalise data ti [0, 1] range

    num_Y_train, height_Y_train, width_Y_train = Y_train.shape
    num_Y_test, height_Y_test, width_Y_test = Y_test.shape

    print('num_Y_train :', num_Y_train, 'height_Y_train :', height_Y_train, 'width_Y_train :', width_Y_train,
          'num_Y_test :', num_Y_test)

    Y_train = Y_train.reshape(num_Y_train, height_Y_train * width_Y_train)
    Y_test = Y_test.reshape(num_Y_test, height_Y_test * width_Y_test)
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    Y_train /= 255  # np.max(Y_train)  # Normalise data to [0, 1] range
    Y_test /= 255  # np.max(Y_test)  # Normalise data to [0, 1] range

    inp = Input(shape=(height_X_train * width_X_train,))
    inp_norm = BatchNormalization(axis=1)(inp)
    hidden1 = Dense(hidden_size, activation='relu')(inp_norm)
    hidden2 = Dense(hidden_size, activation='relu')(hidden1)
    hidden3 = Dense(hidden_size, activation='tanh')(hidden2)
    out = Dense(height_Y_train * width_Y_train, activation='sigmoid')(hidden3) # activation='relu'
    # https://keras.io/activations/
    # activation : relu, elu

    model = Model(inputs=inp, outputs=out)

    # https://keras.io/losses/
    # https://keras.io/optimizers/

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    train_result = model.fit(X_train, Y_train,  # Train the model using the training set...
                             batch_size=batch_size,  # nb_epoch=num_epochs, verbose=0,
                             epochs=num_epochs,
                             verbose=1,
                             validation_split=0.0)  # ...holding out 10% of the data for validation

    test_result = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set

    print_result(train_result, test_result)

    prediction = model.predict(X_test, batch_size=batch_size, verbose=1)

    print()
    # PREDICTION IMAGE
    print('prediction image :')

    # prediction[0].resize((height_Y_test, width_Y_test))
    prediction.resize((num_Y_test, height_Y_test, width_Y_test))
    prediction = np.rint(prediction * 255).astype('uint8')
    # print(prediction[0])
    print_ndarray_info(prediction)  # reshape((3, 4)) => a ; resize((2,6)) => on place
    print_ndarray_info(prediction[0])

    from keras.backend import clear_session
    clear_session()

    from image_handler import get_image
    prediction_image = get_image(prediction[0], mode='L')
    prediction_image.show(title='Prediction 28')
    prediction_image.save('saved_images/prediction.png')

    # ZOOMED OUT IMAGE
    print('zoomed out image :')

    X_test.resize((num_X_test, height_X_test, width_X_test))
    X_test = np.rint(X_test * 255).astype('uint8')
    print_ndarray_info(X_test)
    print_ndarray_info(X_test[0])

    zoomed_out_image = get_image(X_test[0], mode='L')
    # zoomed_out_image.show(title='Zoomed out 14')

    # ORIGINAL IMAGE
    print('original image :')

    Y_test.resize((num_Y_test, height_Y_test, width_Y_test))
    Y_test = np.rint(Y_test * 255).astype('uint8')
    print_ndarray_info(Y_test)
    print_ndarray_info(Y_test[0])

    original_image = get_image(Y_test[0], mode='L')
    original_image.show(title='ORIGINAL IMAGE 28')
    original_image.save('saved_images/original.png')


def test(model, X_test, Y_test, verbose=0):
    print('neural network testing...')
    model.evaluate(X_test, Y_test, verbose=verbose)


def run(data):
    print('neural network running...')

    # list to ndarray : np.array(list) # dtype=np.uint8, dtype=np.float32
    # ndarray to list : ndarray.tolist()





    print('data :', data[:10])

    # mnist_data_file = open('mnist-data.pkl', 'rb')
    # mnist = pickle.load(mnist_data_file)
    # mnist_data_file.close()

    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(type(X_train), type(y_train))

    print('X_train[0] :', X_train[0])
    print('X_train[0][0] :', X_train[0][0])

    print(X_train.ndim, X_train.shape, X_train.size, X_train.dtype, X_train.itemsize)

    # neural_network = shelve.open('neural_network')
    # print(neural_network)
    return X_train[0].tolist()

    return data


def pickle_mnist():
    from keras.datasets import mnist
    mnist_data = mnist.load_data()
    print(mnist_data[:3])
    # (X_train, y_train), (X_test, y_test) = mnist_data
    # print(type(X_train), type(y_train))

    mnist_data_file = open('mnist-data.pkl', 'wb')
    pickle.dump(mnist_data, mnist_data_file)
    mnist_data_file.close()
    # fitted_model = pickle.load(neural_network_model_file)


def print_ndarray_info(ndarray):
    print(ndarray.ndim, ndarray.shape, ndarray.size, ndarray.dtype, ndarray.itemsize)


if __name__ == '__main__':
    print('neural_network module running...')
    # pickle_mnist()
    # run([])
    # fit_dense()
    fit_conv()
    #
    # from keras.datasets import mnist
    #
    # (X_train, y_train), (X_test, y_test) = get_mnist_dataset()  # mnist.load_data()
    #
    # print_ndarray_info(X_train)
    # print_ndarray_info(X_test)
    #
    # print_ndarray_info(y_train)
    # print_ndarray_info(y_test)
    # print('y_train', y_train[0])
    # print('y_test', y_test[0])
    #
    # num_classes = 10
    # from keras.utils import np_utils
    # Y_train = np_utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
    # Y_test = np_utils.to_categorical(y_test, num_classes)
    #
    # print_ndarray_info(Y_train)
    # print_ndarray_info(Y_test)
    # print('Y_train', Y_train)
    # print('Y_test', Y_test)
    #
    #
    # import sqlite3
    #
    # import pickle
    #
    # data = 'asdf'
    # data_file = open('data.pkl', 'wb')
    # pickle.dump(data, data_file)
    # data_file.close()
    #
    # data_file = open('data.pkl', 'rb')
    # pickled_data = pickle.load(data_file)
    # print(pickled_data)
    # data_file.close()
