import shelve
import pickle

import numpy as np
import math
import time

import keras
import keras.metrics as M
import keras.backend as K
from scipy.sparse.data import _data_matrix

from dataset import get_mnist_dataset, get_mnist_dataset_part, get_srcnn_mnist_dataset, get_srcnn_mnist_dataset_part, \
    get_srcnn_rgb_mnist_dataset, get_srcnn_rgb_mnist_dataset_part, get_srcnn_rgb_cifar10_dataset, \
    get_srcnn_rgb_cifar10_dataset_part, get_srcnn_rgb_cifar10_20000_dataset, get_srcnn_rgb_cifar10_20000_dataset_part, \
    get_pasadena_dataset, get_pasadena_dataset_part, get_hundred_dataset, get_hundred_dataset_part, get_dataset_part
import metrics


def print_result(train_result, test_result):
    print()
    print('test_result :', test_result)
    print('train_result :', 'epochs :', train_result.epoch, 'history :', train_result.history)


def print_train_result(train_result):
    print('train_result :', 'epochs :', train_result.epoch, 'history :', train_result.history)


def print_test_result(test_result):
    print()
    print('test_result :', test_result)


def print_shape(name, train, test):
    if len(train.shape) == 3:
        num_train, height_train, width_train = map(str, train.shape)
        num_test, height_test, width_test = map(str, test.shape)
        depth_train = depth_test = 1
    elif len(train.shape) == 4:
        num_train, height_train, width_train, depth_train = map(str, train.shape)
        num_test, height_test, width_test, depth_test = map(str, test.shape)

    print(name + '_train : [ num : ' + num_train, 'height : ' + height_train, 'width : ' + width_train,
          'depth : ' + depth_train + ' ]',
          name + '_test : [ num : ' + num_test, 'height : ' + height_test, 'width : ' + width_test,
          'depth : ' + depth_test + ' ]', sep=', ')


def plot_results(train_result, test_result):
    print('plotting train and test results...')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epoch, history = train_result.epoch, train_result.history
    epoch = [i + 1 for i in epoch]

    # subplots = [221, 222, 223]
    history_keys = sorted(history.keys())
    history_keys.remove('loss')
    test_result_dict = {'loss': test_result[0]}
    test_result_dict.update(zip(history_keys, test_result[1:]))

    plt.title('Train and test results')
    # ro, r--, bs, g^, -, -., :
    plt.figure(1, figsize=(16.0, 10.0))
    # ax = plt.figure(1).gca()  # plt.figure(1)
    for i, (metric, values) in enumerate(history.items()):
        # plt.subplot(subplots[i])
        plt.subplot(221 + i)
        plt.title(metric)
        plt.xlabel('Epoches')
        plt.ylabel('Values')
        # train result
        plt.plot(epoch, values, 'r')
        min_value, max_value, test_result_item = min(values), max(values), test_result_dict[metric]
        dy = 0
        # max train result
        plt.plot(epoch, [max_value] * len(epoch), 'r:')
        plt.text(epoch[-2], max_value + dy, str(round(max_value, 4)))
        # min train result
        plt.plot(epoch, [min_value] * len(epoch), 'r:')
        plt.text(epoch[-2], min_value + dy, str(round(min_value, 4)))
        # test result
        plt.plot(epoch, [test_result_item] * len(epoch), 'b')
        plt.text(epoch[0], test_result_item + dy, str(round(test_result_item, 4)))

    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.subplots_adjust(top=0.9, bottom=0.10, left=0.10, right=0.90, hspace=0.5,
                        wspace=0.4)
    # plt.show()
    print('saving train-and-test-results.png')
    plt.savefig('saved_images/train-and-test-results.png')
    print('saving train-and-test-results.svg')
    plt.savefig('saved_images/train-and-test-results.svg')


def mse_L(y_true, y_pred):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # 0 == ideal, the less the better
    print('MSE metric running...')
    return keras.metrics.mean_squared_error(y_true, y_pred)


def psnr_L(y_true, y_pred):
    # Peak signal-to-noise ratio
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # NORMAL VALUES : [30, 40]
    print('PSNR metric running...')

    maxf = keras.backend.max(y_true)  # 255
    mse = keras.metrics.mean_squared_error(y_true, y_pred)
    arg = maxf / keras.backend.sqrt(mse)
    ten = keras.backend.constant(10)
    return 20 * keras.backend.log(arg) / keras.backend.log(ten)


# import tensorflow as tf
# sess = tf.Session()

def psnr_3_channels(y_true, y_pred):
    # Peak signal-to-noise ratio
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # NORMAL VALUES : [30, 40]
    print('PSNR (3 channels) metric running...')

    maxf = 1.0  # K.max(y_true)
    mse = keras.metrics.mean_squared_error(y_true, y_pred)
    eps = K.constant(1e-6)  # K.epsilon()
    arg = maxf / K.sqrt(mse + eps)  # + K.epsilon()
    ten = K.constant(10)
    return 20 * K.log(arg) / K.log(ten)
    # return 1.0 / K.clip(mse, K.epsilon(), None)

    # import tensorflow as tf
    # v = (tf.Variable(y_true).data).cpu().numpy()
    # print(y_true.get_shape())
    # print(mse.data.numpy())

    a = 1.0 / mse
    # K.set_floatx('float16')
    print(K.floatx())
    return K.log(2.0)

    ten = K.constant(10)
    return 20 * K.log(arg) / K.log(ten)
    #
    # import tensorflow as tf
    # from tensorflow.python.framework import ops
    # from tensorflow.python.ops import gen_math_ops
    # name = None
    # x = tf.constant(5)
    # with ops.name_scope(name, "Square", [x]) as name:
    #     return gen_math_ops.square(x, name=name)
    #
    # result = _op_def_lib.apply_op("Square", x=x, name=name)


    maxf = keras.backend.max(y_true)  # 255
    mse = keras.metrics.mean_squared_error(y_true, y_pred)
    arg = maxf / keras.backend.sqrt(mse)
    ten = keras.backend.constant(10)
    one_tenth = keras.backend.constant(0.0511)
    one = keras.backend.constant(1)
    s = keras.backend.sqrt(mse)
    arg = keras.backend.sqrt(mse)
    # global sess
    # sess = keras.backend.get_session()
    # val = keras.backend.get_value(mse)
    # K.set_epsilon(1e-10)

    import tensorflow as tf
    a = tf.constant(5.0)
    with tf.Session() as sess:
        with sess.as_default():
            print(sess.run(a * 2))
            v = sess.run(a * 2)
            print(type(v))
            print(type(a))
            print(type(mse))

            print(type(sess.run(y_true)))

            return K.constant(v)
    return a
    # sess = tf.InteractiveSession()
    # v = mse.eval()
    # return K.constant(v)

    # with tf.InteractiveSession() as sess:
    # with tf.Session() as sess:
    #     v = mse.eval(session=sess)
    #     return K.constant(v)
    # a = tf.constant(5)
    # return a

    # val = mse.eval(session=sess)
    # a = 20 * np.log10(1.0 / np.sqrt(val))
    # return keras.backend.constant(a)
    # keras.backend.set_epsilon(0.00000001)

    # return keras.backend.round(ten)
    # return keras.backend.greater_equal(one_tenth, ten)
    # return 20 * keras.backend.log(arg) / keras.backend.log(ten)
    return 20 * keras.backend.log(arg) / keras.backend.log(ten)

    # import tensorflow as tf

    # import tensorflow.contrib.metrics as m
    # print(tf.shape(y_true))
    # print(type(res))
    # print(K.max(y_true))
    # print(K.shape(y_true))

    # res = m.streaming_mean_squared_error(y_true, y_pred)
    # print(res)
    # print(type(K.epsilon()))
    # print(type(K.mean(y_pred)))
    # return K.mean(y_pred)


# TODO IFC (Information Fidelity Criterion), NQM (Noise Quality Measure),
# TODO PSNR (weighted peak signal-to-noise ratio),
# TODO MSSSIM (multiscale structure similarity index)

def ssim_3_channels(y_true, y_pred):
    print('SSIM (3 channels) metric running...')

    mx = K.mean(y_true)
    my = K.mean(y_pred)
    return K.constant(-1)


def custom_relu(x):
    return keras.activations.relu(x, max_value=1.0)


def normalise_ndarrrays(ndarrays):
    result = []
    for data in ndarrays:
        result.append(data / np.max(data))
    return np.array(result)


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
                  metrics=['accuracy', psnr_L])

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


def fit_dense_improved():
    print('dense-improved neural network fitting...')

    from keras.layers import Input, Dense, Conv2D, MaxPooling2D
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l2  # L2-regularisation

    batch_size = 128  # in each iteration we consider 128 training examples at once
    num_epochs = 3  # we iterate twenty times over the entire training set
    hidden_size = 512
    kernel_size = 3  # we will use 3x3 kernels throughout
    conv_depth = 32  # use 32 kernels in both convolutional layers
    depth = 1

    (X_train, Y_train), (X_test, Y_test) = get_mnist_dataset_part(train_part=1)

    num_X_train, height_X_train, width_X_train = X_train.shape
    num_X_test, height_X_test, width_X_test = X_test.shape

    # print('num_X_train :', num_X_train, 'height_X_train :', height_X_train, 'width_X_train :', width_X_train,
    #       'num_X_test :', num_X_test)
    print_shape('X', X_train, X_test)

    X_train = X_train.reshape(num_X_train, height_X_train * width_X_train)
    X_test = X_test.reshape(num_X_test, height_X_train * width_X_train)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255  # np.max(X_train)  # Normalise data to [0, 1] range
    X_test /= 255  # np.max(X_test)  # Normalise data ti [0, 1] range

    num_Y_train, height_Y_train, width_Y_train = Y_train.shape
    num_Y_test, height_Y_test, width_Y_test = Y_test.shape

    # print('num_Y_train :', num_Y_train, 'height_Y_train :', height_Y_train, 'width_Y_train :', width_Y_train,
    #       'num_Y_test :', num_Y_test)
    print_shape('Y', Y_train, Y_test)

    Y_train = Y_train.reshape(num_Y_train, height_Y_train * width_Y_train)
    Y_test = Y_test.reshape(num_Y_test, height_Y_test * width_Y_test)
    Y_train = Y_train.astype('float32')
    Y_test = Y_test.astype('float32')
    Y_train /= 255  # np.max(Y_train)  # Normalise data to [0, 1] range
    Y_test /= 255  # np.max(Y_test)  # Normalise data to [0, 1] range

    use_saved = False
    if use_saved:
        # returns a compiled model
        # identical to the previous one
        model = keras.models.load_model('models/dense-improved.h5', custom_objects={'psnr_L': psnr_L})
    else:
        inp = Input(shape=(height_X_train * width_X_train,))
        inp_norm = BatchNormalization(axis=1)(inp)
        hidden1 = Dense(hidden_size, activation='relu')(inp_norm)  # 'relu'
        hidden2 = Dense(hidden_size, activation='relu')(hidden1)  # 'relu'
        hidden3 = Dense(hidden_size, activation='sigmoid')(hidden2)  # 'tanh'
        out = Dense(height_Y_train * width_Y_train, activation='sigmoid')(hidden3)  # activation='sigmoid'
        # https://keras.io/activations/
        # activation : relu, elu

        model = Model(inputs=inp, outputs=out)

        # https://keras.io/losses/
        # https://keras.io/optimizers/

        model.compile(loss='mean_squared_error',
                      optimizer='nadam',  # 'RMSprop', 'adam'
                      metrics=['accuracy', psnr_L, mse_L])

        train_result = model.fit(X_train, Y_train,  # Train the model using the training set...
                                 batch_size=batch_size,  # nb_epoch=num_epochs, verbose=0,
                                 epochs=num_epochs,
                                 verbose=1,
                                 validation_split=0.0)  # ...holding out 10% of the data for validation

        print_train_result(train_result)

        # saving the model

        # model.save('models/dense-improved.h5')  # creates a HDF5 file 'models/dense-improved.h5'
        # del model  # deletes the existing model

    # getting results

    test_result = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set

    print_test_result(test_result)

    if not use_saved:
        plot_results(train_result, test_result)

    prediction = model.predict(X_test, batch_size=batch_size, verbose=1)

    show_images = False
    for i in range(10):
        print()
        # PREDICTION IMAGE
        print('prediction image ' + str(i) + ' :')

        # prediction[0].resize((height_Y_test, width_Y_test))
        prediction.resize((num_Y_test, height_Y_test, width_Y_test))
        prediction = np.rint(prediction * 255).astype('uint8')
        # print(prediction[0])
        print_ndarray_info(prediction)  # reshape((3, 4)) => a ; resize((2,6)) => on place
        print_ndarray_info(prediction[i])

        from keras.backend import clear_session
        clear_session()

        from image_handler import get_image
        prediction_image = get_image(prediction[i], mode='L')
        if show_images:
            prediction_image.show(title='Prediction 28')
        prediction_image.save('saved_images/' + str(i) + '_prediction.png')

        # ZOOMED OUT IMAGE
        print('zoomed out image ' + str(i) + ':')

        X_test.resize((num_X_test, height_X_test, width_X_test))
        X_test = np.rint(X_test * 255).astype('uint8')
        print_ndarray_info(X_test)
        print_ndarray_info(X_test[i])

        zoomed_out_image = get_image(X_test[i], mode='L')
        # zoomed_out_image.show(title='Zoomed out 14')

        # ORIGINAL IMAGE
        print('original image ' + str(i) + ':')

        Y_test.resize((num_Y_test, height_Y_test, width_Y_test))
        Y_test = np.rint(Y_test * 255).astype('uint8')
        print_ndarray_info(Y_test)
        print_ndarray_info(Y_test[i])

        original_image = get_image(Y_test[i], mode='L')
        if show_images:
            original_image.show(title='ORIGINAL IMAGE 28')
        original_image.save('saved_images/' + str(i) + '_original.png')


def fit_conv():
    print('neural network fitting...')

    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l2  # L2-regularisation

    batch_size = 512  # 512 #128  # in each iteration we consider 128 training examples at once
    num_epochs = 20  # we iterate twenty times over the entire training set
    l2_lambda = 0.0001  # use 0.0001 as a L2-regularisation factor
    kernel_size = 3  # 3  # we will use 3x3 kernels throughout
    pool_size = 2  # we will use 2x2 pooling throughout
    conv_depth = 32  # use 32 kernels in both convolutional layers
    drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
    hidden_size = 512  # 128
    depth = 1

    (X_train, Y_train), (X_test, Y_test) = get_mnist_dataset_part()  # train_part=0.2)

    num_X_train, height_X_train, width_X_train = X_train.shape
    num_X_test, height_X_test, width_X_test = X_test.shape

    print('num_X_train :', num_X_train, 'height_X_train :', height_X_train, 'width_X_train :', width_X_train,
          'num_X_test :', num_X_test)

    # print(X_train[0])
    X_train = X_train.reshape(num_X_train, depth, height_X_train, width_X_train)
    # print('sdf')
    # print(X_train.shape)
    # print(X_train[0])
    # exit()

    X_test = X_test.reshape(num_X_test, depth, height_X_test, width_X_test)
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

    from keras import backend
    backend.set_image_dim_ordering('th')

    inp = Input(shape=(depth, height_X_train, width_X_train))
    # inp = Input(shape=(depth, height_X_train, width_X_train))
    inp_norm = BatchNormalization(axis=1)(inp)
    conv_1 = Conv2D(conv_depth, (kernel_size, kernel_size), padding='same', kernel_initializer='he_uniform',
                    # 'he_uniform'
                    kernel_regularizer=l2(l2_lambda), activation='relu')(inp_norm)  # 'relu' sigmoid
    conv_1 = BatchNormalization(axis=1)(conv_1)
    conv_2 = Conv2D(conv_depth, (kernel_size, kernel_size), padding='same', kernel_initializer='he_uniform',
                    # 'he_uniform',
                    kernel_regularizer=l2(l2_lambda), activation='relu')(conv_1)  # 'relu'
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    flat = Flatten()(drop_1)
    # hidden = Dense(hidden_size, kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_lambda), activation='relu')(
    #     flat)
    hidden = Dense(hidden_size, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda), activation='relu')(
        flat)
    hidden = BatchNormalization(axis=1)(hidden)
    drop_2 = Dropout(drop_prob_2)(hidden)
    out = Dense(height_Y_train * width_Y_train, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda),
                activation='relu')(
        drop_2)  # activation='relu'
    # out = Dense(10, init='glorot_uniform', W_regularizer=l2(l2_lambda), activation='relu')(inp)  # (hidden)  # activation='relu'

    # hidden1 = Dense(hidden_size, activation='relu')(inp_norm)
    # hidden2 = Dense(hidden_size, activation='relu')(hidden1)
    # hidden3 = Dense(hidden_size, activation='tanh')(hidden2)
    # out = Dense(height_Y_train * width_Y_train, activation='sigmoid')(hidden3)  # activation='relu'
    # https://keras.io/activations/
    # activation : relu, elu

    model = Model(inputs=inp, outputs=out)

    from keras.utils import plot_model
    plot_model(model, to_file='saved_images/model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    # https://keras.io/losses/
    # https://keras.io/optimizers/

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy', psnr_L])

    print(model.summary())
    # exit()

    train_result = model.fit(X_train, Y_train,  # Train the model using the training set...
                             batch_size=batch_size,  # nb_epoch=num_epochs, verbose=0,
                             epochs=num_epochs,
                             verbose=1,
                             validation_split=0.0)  # ...holding out 10% of the data for validation

    test_result = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set

    print_result(train_result, test_result)

    prediction = model.predict(X_test, batch_size=batch_size, verbose=1)

    print(model.summary())

    show_images = False
    for i in range(10):
        print()
        # PREDICTION IMAGE
        print('prediction image ' + str(i) + ' :')

        # prediction[0].resize((height_Y_test, width_Y_test))
        prediction.resize((num_Y_test, height_Y_test, width_Y_test))
        prediction = np.rint(prediction * 255).astype('uint8')
        # print(prediction[0])
        print_ndarray_info(prediction)  # reshape((3, 4)) => a ; resize((2,6)) => on place
        print_ndarray_info(prediction[i])

        from keras.backend import clear_session
        clear_session()

        from image_handler import get_image
        prediction_image = get_image(prediction[i], mode='L')
        if show_images:
            prediction_image.show(title='Prediction 28')
        prediction_image.save('saved_images/' + str(i) + '_prediction.png')

        # ZOOMED OUT IMAGE
        print('zoomed out image ' + str(i) + ':')

        # X_test.resize((num_X_test, height_X_test, width_X_test))
        # X_test = np.rint(X_test * 255).astype('uint8')
        # print_ndarray_info(X_test)
        # print_ndarray_info(X_test[i])
        #
        # zoomed_out_image = get_image(X_test[i], mode='L')

        # zoomed_out_image.show(title='Zoomed out 14')

        # ORIGINAL IMAGE
        print('original image ' + str(i) + ':')

        Y_test.resize((num_Y_test, height_Y_test, width_Y_test))
        Y_test = np.rint(Y_test * 255).astype('uint8')
        print_ndarray_info(Y_test)
        print_ndarray_info(Y_test[i])

        original_image = get_image(Y_test[i], mode='L')
        if show_images:
            original_image.show(title='ORIGINAL IMAGE 28')
        original_image.save('saved_images/' + str(i) + '_original.png')


def fit_conv_improved():
    print('SRCNN running...')

    from keras.layers import Input, Conv1D, Conv2D, Conv3D
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization  # batch normalisation
    from keras.regularizers import l2  # L2-regularisation
    from keras.constraints import min_max_norm, max_norm
    from keras.activations import relu

    f_1 = 9
    f_2 = 1
    f_3 = 5
    n_1 = 64
    n_2 = 32
    c = 1

    batch_size = 64  # 128  # in each iteration we consider 128 training examples at once
    num_epochs = 2  # we iterate twenty times over the entire training set
    kernel_size = 3  # f_1, f_2, f_3
    conv_depth = 32  # n_1, n_2
    depth = 1  # c
    l2_lambda = 0.0001

    use_rgb_dataset = True
    if use_rgb_dataset:
        depth = c = 3
        # (X_train, Y_train), (X_test, Y_test) = get_srcnn_rgb_mnist_dataset_part(train_part=0.1, test_part=0.1)
        # (X_train, Y_train), (X_test, Y_test) = get_srcnn_rgb_cifar10_dataset_part(train_part=1, test_part=1)
        (X_train, Y_train), (X_test, Y_test) = get_srcnn_rgb_cifar10_20000_dataset_part(train_part=0.1, test_part=0.1)
        dataset_name = 'CIFAR-10 20000'
        # (X_train, Y_train), (X_test, Y_test) = get_pasadena_dataset_part(train_part=0.5, test_part=0.5)
        # (X_train, Y_train), (X_test, Y_test) = get_hundred_dataset_part(train_part=1, test_part=1)
        # dataset_name = 'HUNDRED 40'

        num_X_train, height_X_train, width_X_train, _ = X_train.shape
        num_X_test, height_X_test, width_X_test, _ = X_test.shape
        print_shape('X', X_train, X_test)

        num_Y_train, height_Y_train, width_Y_train, _ = Y_train.shape
        num_Y_test, height_Y_test, width_Y_test, _ = Y_test.shape
        print_shape('Y', Y_train, Y_test)

        print('')
    else:
        depth = c = 1
        (X_train, Y_train), (X_test, Y_test) = get_srcnn_mnist_dataset_part(train_part=0.1, test_part=1)

        num_X_train, height_X_train, width_X_train = X_train.shape
        num_X_test, height_X_test, width_X_test = X_test.shape
        print_shape('X', X_train, X_test)

        num_Y_train, height_Y_train, width_Y_train = Y_train.shape
        num_Y_test, height_Y_test, width_Y_test = Y_test.shape
        print_shape('Y', Y_train, Y_test)
    # print(np.max(X_train), np.max(Y_train))
    # print(np.max(X_test), np.max(Y_test))

    # K.set_floatx('float64')
    dataset_type = 'float32'
    # print(K.floatx())

    X_train = X_train.astype(dataset_type)
    X_test = X_test.astype(dataset_type)
    Y_train = Y_train.astype(dataset_type)
    Y_test = Y_test.astype(dataset_type)

    X_train /= 255  # np.max(X_train)  # Normalise data to [0, 1] range
    X_test /= 255  # np.max(X_test)  # Normalise data ti [0, 1] range
    Y_train /= 255  # np.max(Y_train)  # Normalise data to [0, 1] range
    Y_test /= 255  # np.max(Y_test)  # Normalise data to [0, 1] range

    # print(X_train[0])
    # print(X_train.shape)
    # exit()
    X_train = X_train.reshape(num_X_train, depth, height_X_train, width_X_train)
    X_test = X_test.reshape(num_X_test, depth, height_X_test, width_X_test)
    Y_train = Y_train.reshape(num_Y_train, depth, height_Y_train, width_Y_train)  # , height_Y_train * width_Y_train)
    Y_test = Y_test.reshape(num_Y_test, depth, height_Y_test, width_Y_test)  # , height_Y_test * width_Y_test)

    # print(np.max(X_train), np.max(Y_train))
    # print(np.max(X_test), np.max(Y_test))
    # making PSNR metric
    psnr_3_callback = metrics.MetricsCallbackPSNR(X=(X_train, X_test), Y=(Y_train, Y_test), batch_size=batch_size)
    min_max_callback = metrics.MetricsCallbackMinMax(X=(X_train, X_test), Y=(Y_train, Y_test), batch_size=batch_size)
    if not use_rgb_dataset:
        ssim_3_callback = metrics.MetricsCallbackSSIM(X=(X_train, X_test), Y=(Y_train, Y_test), batch_size=batch_size,
                                                      mode='L')

    use_saved = False
    # path_to_model = 'models/srcnn-cifar10-20000-9_3_1_5-64_32_32-20epochs-he_uniform-custom_relu.h5'
    path_to_model = 'results/' + dataset_name + '/model/' + \
                    'srcnn-cifar10-20000-9_3_1_5-64_32_32-20epochs-he_uniform-custom_relu.h5'
    if use_saved:
        # returns a compiled model
        # identical to the previous one
        model = keras.models.load_model(path_to_model, custom_objects={'psnr_L': psnr_L})
    else:
        print('SRCNN fitting...')
        from keras import backend
        backend.set_image_dim_ordering('th')

        # TODO kerner_initializer='he_uniform'

        # f_1-f_2-f_3
        # 9-1-5
        # 9-3-5
        # 9-5-5
        # 9-1-1-5
        # 9-3-1-5
        # 9-5-1-5

        # 64, 2 | 9, 3, 1, 5 | 32, 16, 16 | [24.4]

        batch_size = 64  # 128  # in each iteration we consider 128 training examples at once
        num_epochs = 2
        f_1, f_2, f_2_2, f_3 = 9, 3, 1, 5  # 9, 3, 1, 5 (32, 16, 16) [24.4]
        n_1, n_2, n_2_2 = 64, 32, 32  # 32, 16, 16 # 64, 32, 32  #

        inp = Input(shape=(c, height_X_train, width_X_train))

        # inp_norm = BatchNormalization(axis=1)(inp)

        conv_1 = Conv2D(n_1, (f_1, f_1), padding='same', activation='relu', kernel_regularizer=l2(l2_lambda),
                        kernel_initializer='he_uniform')(inp)

        # conv_1 = BatchNormalization(axis=1)(conv_1)

        conv_2 = Conv2D(n_2, (f_2, f_2), padding='same', activation='relu', kernel_regularizer=l2(l2_lambda),
                        kernel_initializer='he_uniform')(conv_1)

        conv_2 = Conv2D(n_2_2, (f_2_2, f_2_2), padding='same', activation='relu', kernel_regularizer=l2(l2_lambda),
                        kernel_initializer='he_uniform')(  # custom_relu
            conv_2)

        # conv_2 = BatchNormalization(axis=1)(conv_2)

        # conv_3 = Conv2D(c, (f_3, f_3), padding='same', activation=custom_relu, kernel_regularizer=l2(l2_lambda),
        #                 kernel_constraint=max_norm(1.0))(conv_2)
        conv_3 = Conv2D(c, (f_3, f_3), padding='same', activation=custom_relu, kernel_regularizer=l2(l2_lambda), )(
            conv_2)
        # kernel_constraint=max_norm(1.0))(conv_2)

        # conv_1 = Conv2D(n_1, (f_1, f_1), padding='same', activation='relu')(inp)#, kernel_regularizer=l2(l2_lambda))(inp)
        #
        # # conv_1 = BatchNormalization(axis=1)(conv_1)
        #
        # conv_2 = Conv2D(n_2, (f_2, f_2), padding='same', activation='relu')(conv_1)#, kernel_regularizer=l2(l2_lambda))(conv_1)
        #
        # conv_3 = Conv2D(c, (f_3, f_3), padding='same', activation='relu')(conv_2)#, kernel_regularizer=l2(l2_lambda))(conv_2)

        # conv_1 = Conv3D(n_1, (c, f_1, f_1), activation='relu', input_shape=(c, height_X_train, width_X_train))

        # conv_1 = Conv3D(n_1, (c, f_1, f_1), activation='relu')(inp)
        #
        # conv_2 = Conv3D(n_2, (n_1, f_2, f_2), activation='relu')(conv_1)
        #
        # conv_3 = Conv3D(c, (n_2, f_3, f_3), activation='relu')(conv_2)
        #
        out = conv_3

        # creating model
        model = Model(inputs=inp, outputs=out)

        from keras.utils import plot_model
        plot_model(model, to_file='results/' + dataset_name + '/model/SRCNN-model.png', show_shapes=True,
                   show_layer_names=True, rankdir='TB')
        plot_model(model, to_file='results/' + dataset_name + '/model/SRCNN-model.svg', show_shapes=True,
                   show_layer_names=True, rankdir='TB')

        # keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, decay=? 1e-6,
        # momentum=? 0.9, nesterov=True)
        nadam = keras.optimizers.Nadam()  # lr=0.002, clipnorm=1.0, clipvalue=1.0)  # clipvalue=0.5 [-0.5, 0.5]

        model.compile(loss='mean_squared_error',
                      optimizer=nadam,  # 'nadam',
                      metrics=['accuracy',
                               # M.mean_absolute_error,
                               # M.categorical_accuracy,
                               # M.binary_accuracy,
                               # M.top_k_categorical_accuracy,
                               # M.mean_squared_logarithmic_error, # +++
                               M.mean_squared_error,
                               # mse_L
                               # ssim_3_channels,
                               psnr_3_channels
                               ])
        # #, psnr_3_channels])

        print(model.summary())

        train_result = model.fit(X_train, Y_train,  # Train the model using the training set...
                                 batch_size=batch_size,  # nb_epoch=num_epochs, verbose=0,
                                 epochs=num_epochs,
                                 verbose=1,
                                 validation_split=0.0,  # ...holding out 10% of the data for validation
                                 callbacks=[
                                     psnr_3_callback,
                                     # ssim_3_callback,
                                     min_max_callback,
                                 ])

        print_train_result(train_result)

        # exit()
        # saving the model
        # path = 'models/srcnn-cifar10-20000-9_3_1_5-64_32_32-20epochs-he_uniform-custom_relu.h5'

        # print('saving model to ' + path_to_model + '...')
        # model.save(path_to_model)
        # print('model ' + path_to_model + ' saved')

        # creates a HDF5 file 'models/dense-improved.h5'
        # del model  # deletes the existing model

    # getting results
    test_result = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set
    print_test_result(test_result)

    # making plot
    if not use_saved:
        plot_results(train_result, test_result)

    print('PSNR history :', psnr_3_callback.history)
    print('PSNR epoch history : ', psnr_3_callback.epoch_history)

    if not use_rgb_dataset:
        print('SSIM history :', ssim_3_callback.history)
        print('SSIM epoch history : ', ssim_3_callback.epoch_history)

    print('MIN-MAX history :', min_max_callback.history)
    print('MIN-MAX epoch history : ', min_max_callback.epoch_history)

    prediction = model.predict(X_test, batch_size=batch_size, verbose=1)
    print(model.summary())

    from keras.backend import clear_session
    clear_session()

    normalise_prediction = True
    if normalise_prediction:
        prediction = normalise_ndarrrays(prediction)

    show_images = False
    for i in range(10):
        print()
        # PREDICTION IMAGE
        print('prediction image ' + str(i) + ' :')

        # prediction[0].resize((height_Y_test, width_Y_test))
        if use_rgb_dataset:
            prediction.resize((num_Y_test, height_Y_test, width_Y_test, depth))
        else:
            prediction.resize((num_Y_test, height_Y_test, width_Y_test))
        prediction = np.rint(prediction * 255).astype('uint8')
        # print(prediction[0])
        print_ndarray_info(prediction)  # reshape((3, 4)) => a ; resize((2,6)) => on place
        print_ndarray_info(prediction[i])

        from image_handler import get_image
        if use_rgb_dataset:
            prediction_image = get_image(prediction[i], mode='RGB')
        else:
            prediction_image = get_image(prediction[i], mode='L')
        if show_images:
            prediction_image.show(title='Prediction 28')
        prediction_image.save('results/' + dataset_name + '/test/' + str(i) + '_prediction.png')

        # ZOOMED OUT IMAGE
        print('zoomed out image ' + str(i) + ':')

        if use_rgb_dataset:
            X_test.resize((num_X_test, height_X_test, width_X_test, depth))
        else:
            X_test.resize((num_X_test, height_X_test, width_X_test))
        X_test = np.rint(X_test * 255).astype('uint8')
        print_ndarray_info(X_test)
        print_ndarray_info(X_test[i])

        if use_rgb_dataset:
            zoomed_out_image = get_image(X_test[i], mode='RGB')
        else:
            zoomed_out_image = get_image(X_test[i], mode='L')
        # zoomed_out_image.show(title='Zoomed out 14')

        # ORIGINAL IMAGE
        print('original image ' + str(i) + ':')

        if use_rgb_dataset:
            Y_test.resize((num_Y_test, height_Y_test, width_Y_test, depth))
        else:
            Y_test.resize((num_Y_test, height_Y_test, width_Y_test))
        Y_test = np.rint(Y_test * 255).astype('uint8')
        print_ndarray_info(Y_test)
        print_ndarray_info(Y_test[i])

        if use_rgb_dataset:
            original_image = get_image(Y_test[i], mode='RGB')
        else:
            original_image = get_image(Y_test[i], mode='L')
        if show_images:
            original_image.show(title='ORIGINAL IMAGE 28')
        original_image.save('results/' + dataset_name + '/test/' + str(i) + '_original.png')

        from image_handler import get_images_difference_metrics
        psnr, ssim, mse = get_images_difference_metrics(original_image, prediction_image)
        psnr, ssim, mse = round(psnr, 4), round(ssim, 4), round(mse, 4)
        print('PSNR : ' + str(psnr), 'SSIM : ' + str(ssim), 'MSE : ' + str(mse), sep=', ')


        # inp = Input(shape=(depth, height_X_train, width_X_train))
        # # inp = Input(shape=(depth, height_X_train, width_X_train))
        # inp_norm = BatchNormalization(axis=1)(inp)
        # conv_1 = Conv2D(conv_depth, (kernel_size, kernel_size), padding='same', kernel_initializer='he_uniform',
        #                 # 'he_uniform'
        #                 kernel_regularizer=l2(l2_lambda), activation='relu')(inp_norm)  # 'relu' sigmoid
        # conv_1 = BatchNormalization(axis=1)(conv_1)
        # conv_2 = Conv2D(conv_depth, (kernel_size, kernel_size), padding='same', kernel_initializer='he_uniform',
        #                 # 'he_uniform',
        #                 kernel_regularizer=l2(l2_lambda), activation='relu')(conv_1)  # 'relu'
        # pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
        # drop_1 = Dropout(drop_prob_1)(pool_1)
        # flat = Flatten()(drop_1)
        # # hidden = Dense(hidden_size, kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_lambda), activation='relu')(
        # #     flat)
        # hidden = Dense(hidden_size, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda), activation='relu')(
        #     flat)
        # hidden = BatchNormalization(axis=1)(hidden)
        # drop_2 = Dropout(drop_prob_2)(hidden)
        # out = Dense(height_Y_train * width_Y_train, kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda),
        #             activation='relu')(
        #     drop_2)  # activation='relu'
        # out = Dense(10, init='glorot_uniform', W_regularizer=l2(l2_lambda), activation='relu')(inp)  # (hidden)  # activation='relu'

        # hidden1 = Dense(hidden_size, activation='relu')(inp_norm)
        # hidden2 = Dense(hidden_size, activation='relu')(hidden1)
        # hidden3 = Dense(hidden_size, activation='tanh')(hidden2)
        # out = Dense(height_Y_train * width_Y_train, activation='sigmoid')(hidden3)  # activation='relu'
        # https://keras.io/activations/
        # activation : relu, elu

        # model = Model(inputs=inp, outputs=out)


def test(model, X_test, Y_test, verbose=0):
    print('neural network testing...')
    model.evaluate(X_test, Y_test, verbose=verbose)


def run(image_data):
    print('neural network running...')

    # preparing image_data
    test = np.array([image_data])
    test = test.astype('float32')
    test /= 255  # np.max(Y_test)  # Normalise data to [0, 1] range
    test = test.reshape(1, 3, 32, 32)

    # loading model
    path_to_model = 'models/srcnn-cifar10-20000-9_3_1_5-64_32_32-20epochs-he_uniform-custom_relu.h5'
    model = keras.models.load_model(path_to_model, custom_objects={
        'psnr_3_channels': psnr_3_channels,
        'custom_relu': custom_relu,
    })  # , custom_objects={'psnr_L': psnr_L})
    print(model.summary())

    batch_size = 64
    prediction = model.predict(test, batch_size=batch_size, verbose=1)

    from keras.backend import clear_session
    clear_session()

    prediction.resize((1, 32, 32, 3))
    prediction = np.rint(prediction * 255).astype('uint8')
    print_ndarray_info(prediction, verbose=True)  # reshape((3, 4)) => a ; resize((2,6)) => on place
    prediction_image_data = prediction[0]
    print_ndarray_info(prediction_image_data, verbose=True)

    from image_handler import get_image
    prediction_image = get_image(prediction_image_data, mode='RGB')
    # prediction_image.show()

    return prediction_image_data

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


def print_ndarray_info(ndarray, verbose=False):
    if verbose:
        print('ndim :', ndarray.ndim, ' shape :', ndarray.shape, ' size :', ndarray.size, ' dtype :', ndarray.dtype,
              ' itemsize :', ndarray.itemsize)
    else:
        print(ndarray.ndim, ndarray.shape, ndarray.size, ndarray.dtype, ndarray.itemsize)


def handle_time_range(secs):
    msecs, secs = math.modf(secs)
    msecs = int(msecs * 1000)
    secs = int(secs)
    mins = secs // 60
    secs %= 60
    return mins, secs, msecs


def convert_handled_time_range_to_str(handled_time_range):
    mins, secs, msecs = handled_time_range
    return '{} m {} s {} ms'.format(mins, secs, msecs)


if __name__ == '__main__':
    print('neural_network module running...')
    begin_time = time.time()
    # pickle_mnist()
    # run([])
    # fit_dense()  # psnr_L : 19.3015 (19.8356)
    # fit_dense_improved()  # psnr_L : [adam] 21.6834 (22.2798), [nadam] 21.8307 (22.6945)
    # fit_conv()  # psnr_L : [10000] 14.6536 (16.3086) [54000] 19.6423 (18.1249)
    #
    fit_conv_improved()

    print('SRCNN running lasted', convert_handled_time_range_to_str(handle_time_range(time.time() - begin_time)))


    # from keras.utils import normalize
    # a = np.array([1,2,3,4,5])
    # print(normalize(a, axis=0, order=20))
    # print(normalize(a, axis=1, order=2))
    # print(normalize(a, axis=1, order=2))

    # dataset = get_mnist_dataset()
    # print(type(dataset))

    # history = {'acc': [0.013018518519346361, 0.013277777779433462, 0.014203703704531546],
    #           'psnr_L': [16.38225281326859, 19.235429208260996, 20.153424346358687],
    #          'loss': [0.027831019085314539, 0.013503529415914307, 0.010932281403077974]}


    # convert_mnist_dataset_to_ndarrays()

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
