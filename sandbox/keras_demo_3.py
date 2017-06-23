from keras.datasets import mnist  # subroutines for fetching the MNIST dataset
from keras.models import Model  # basic class for specifying and training a neural network
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout, merge
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values
from keras.regularizers import l2  # L2-regularisation
from keras.layers.normalization import BatchNormalization  # batch normalisation
from keras.preprocessing.image import ImageDataGenerator  # data augmentation
from keras.callbacks import EarlyStopping  # early stopping

from keras.backend import clear_session

batch_size = 128  # in each iteration, we consider 128 training examples at once
num_epochs = 50  # we iterate at most fifty times over the entire training set
kernel_size = 3  # we will use 3x3 kernels throughout
pool_size = 2  # we will use 2x2 pooling throughout
conv_depth = 32  # use 32 kernels in both convolutional layers
drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
hidden_size = 128  # there will be 128 neurons in both hidden layers
l2_lambda = 0.0001  # use 0.0001 as a L2-regularisation factor
ens_models = 3  # we will train three separate models on the data

num_train = 60000  # there are 60000 training examples in MNIST
num_test = 10000  # there are 10000 test examples in MNIST

height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
num_classes = 10  # there are 10 classes (1 per digit)

(X_train, y_train), (X_test, y_test) = mnist.load_data()  # fetch MNIST data

# print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], depth, height, width)
# print(X_train.shape)
# exit()
X_test = X_test.reshape(X_test.shape[0], depth, height, width)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = np_utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes)  # One-hot encode the labels

# Explicitly split the training and validation sets
temp = True
if temp:
    num_epochs = 1
    train_size = 500  # 1000  # 54000
    size = 550  # 1100
    X_train = X_train[:size]
    Y_train = Y_train[:size]
    num_test = 1000
    X_test = X_test[:num_test]
    Y_test = Y_test[:num_test]
else:
    train_size = 54000
X_val = X_train[train_size:]
Y_val = Y_train[train_size:]
X_train = X_train[:train_size]
Y_train = Y_train[:train_size]

# TEMP BELOW
import shelve
import pickle

neural_network = shelve.open('neural_network')
neural_network.clear()
# print(neural_network.get('model', 'nothing'))
# neural_network.clear()
# model = neural_network.get('model', None)
if False and not (neural_network.get('model', None) is None):
    print('model is fitted and ready for evaluation')
    print(neural_network['model'])

    neural_network_model_file = open('neural-network-model.pkl', 'rb')
    fitted_model = pickle.load(neural_network_model_file)
    print(fitted_model)
    neural_network_model_file.close()
    exit()

    fitted_model = neural_network['model']
    fitted_model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
    exit()

# TEMP ABOVE

from keras import backend

backend.set_image_dim_ordering('th')

inp = Input(shape=(depth, height, width))  # N.B. Keras expects channel dimension first
inp_norm = BatchNormalization(axis=1)(inp)  # Apply BN to the input (N.B. need to rename here)

outs = []  # the list of ensemble outputs
for i in range(ens_models):
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer), applying BN in between
    conv_1 = Convolution2D(conv_depth, kernel_size, kernel_size, border_mode='same', init='he_uniform',
                           W_regularizer=l2(l2_lambda), activation='relu')(inp_norm)

    conv_1 = BatchNormalization(axis=1)(conv_1)
    conv_2 = Convolution2D(conv_depth, kernel_size, kernel_size, border_mode='same', init='he_uniform',
                           W_regularizer=l2(l2_lambda), activation='relu')(conv_1)
    conv_2 = BatchNormalization(axis=1)(conv_2)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    flat = Flatten()(drop_1)
    hidden = Dense(hidden_size, init='he_uniform', W_regularizer=l2(l2_lambda), activation='relu')(
        flat)  # Hidden ReLU layer
    hidden = BatchNormalization(axis=1)(hidden)
    drop = Dropout(drop_prob_2)(hidden)
    outs.append(Dense(num_classes, init='glorot_uniform', W_regularizer=l2(l2_lambda), activation='softmax')(
        drop))  # Output softmax layer

out = merge(outs, mode='ave')  # average the predictions to obtain the final output

model = Model(input=inp, output=out)  # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer='adam',  # using the Adam optimiser
              metrics=['accuracy'])  # reporting the accuracy

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB') # 'LR' => horizontal
print(model.summary())
exit()

datagen = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)
datagen.fit(X_train)

# fit the model on the batches generated by datagen.flow()---most parameters similar to model.fit


model.fit_generator(datagen.flow(X_train, Y_train,
                                 batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    epochs=num_epochs,  # nb_epoch=num_epochs
                    validation_data=(X_val, Y_val),
                    verbose=1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5)])  # adding early stopping

print('model is fitted and ready for evaluation')
print('model saving...')
neural_network['model'] = True
neural_network_model_file = open('neural-network-model.pkl', 'wb')
# pickle.dump(model, neural_network_model_file)
print('model was saved...')
neural_network.close()

model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!

prediction = model.predict(X_test, batch_size=batch_size, verbose=1)

clear_session()


# print('Y_test :', Y_test[:10])
# print('prediction :', prediction[:10])

Y = Y_test.round()[:10]
P = prediction.round()[:10]
print('Y_test :', Y)
print('prediction :', P)
print('EQUAL?', Y == P)

from keras.utils import serialize_keras_object
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB') # 'LR' => horizontal
ser = serialize_keras_object(model)
# file = open('model.ser', 'w')
# file.write(ser)

from keras.utils.vis_utils import model_to_dot
model_to_dot(model).create(prog='dot', format='svg')