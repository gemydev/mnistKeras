from tensorflow.keras.utils import to_categorical
from keras import layers
from keras import models
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
# add 2 layers with ( tanh , sigmoid ) activation functions
network.add(layers.Dense(8, activation='sigmoid', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='tanh', input_shape=(28 * 28,)))

# change optimizer from rmsprop to adam
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# change epochs=9 , batch_size=10
network.fit(train_images, train_labels, epochs=9, batch_size=10)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
