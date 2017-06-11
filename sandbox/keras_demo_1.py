from keras.datasets import mnist  # subroutines for fetching the MNIST dataset
from keras.models import Model  # basic class for specifying and training a neural network
from keras.layers import Input, Dense  # the two types of neural network layer we will be using
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values
from keras.backend import clear_session

batch_size = 128  # in each iteration we consider 128 training examples at once
# num_epochs = 20  # we iterate twenty times over the entire training set
num_epochs = 1
hidden_size = 512  # there will be 512 neurons in both hidden layers

num_train = 60000  # there are 60000 training examples in MNIST
num_test = 10000  # there are 10000 test examples in MNIST

height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
num_classes = 10  # there are 10 classes (1 per digit)

(X_train, y_train), (X_test, y_test) = mnist.load_data()  # fetch MNIST data

X_train = X_train.reshape(num_train, height * width)  # Flatten data to 1D
X_test = X_test.reshape(num_test, height * width)  # Flatten data to 1D
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255  # Normalise data to [0, 1] range
X_test /= 255  # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes)  # One-hot encode the labels

# forming the layers
inp = Input(shape=(height * width,))  # Our input is a 1D vector of size 784
hidden1 = Dense(hidden_size, activation='relu')(inp)  # First hidden ReLU layer
hidden2 = Dense(hidden_size, activation='relu')(hidden1)  # Second hidden ReLU layer
out = Dense(num_classes, activation='softmax')(hidden2)  # Output softmax layer

model = Model(inputs=inp, outputs=out)  # To define a model just specify its input and output layers

# We use cross-entropy function because it maximizes model confidence in right definition of class
# and it do not care about probability distribution of example getting into another class.
# Функция перекрестной энтропии предназначена для максимизации уверенности модели в правильном определении класса,
# и ее не заботит распределение вероятностей попадания образца в другие классы.
# Функция квадратичной ошибки стремится к тому, чтобы вероятность попадания
# в остальные классы была как можно ближе к нулю.

model.compile(loss='categorical_crossentropy',  # using cross-entropy loss function
              optimizer='adam',  # using the Adam optimiser
              metrics=['accuracy'])  # reporting the accuracy

# running the algorithm

train_result = model.fit(X_train, Y_train,  # Train the model using the training set...
                    batch_size=batch_size,  # nb_epoch=num_epochs, verbose=0,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=0.1)  # ...holding out 10% of the data for validation
test_result = model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set

print()
print('test_result :', test_result)
print('train_result :', 'epochs :', train_result.epoch, 'history :', train_result.history)

clear_session()
