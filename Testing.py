from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print(train_data[0])

len(train_data[0]), len(train_data[1])

# adding a dictionary that maps words to an integer index
word_index = imdb.get_word_index()
# first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["UNKNOWN"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, "?") for i in text])


decode_review(train_data[0])
# training the data with the keras
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
# testing the data after we train it
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
len(train_data[0]), len(train_data[1])
print(train_data[0])

# input shape is the vocab count used for the reviews which is 10,000 words
vocab_size = 10000
# here we will add our layers for testing purposes
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# now since we have completed our model we will compile it next
model.compile(optimizer='adam',
              loss='binary_entropy',
              metrics=['accuracy'])

# this will act as our validation set
# training data for the x value
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

# training labels for the y value
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# now we will train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    validation_data=(x_val, y_val),
                    verbose=1)

# evaluating the model
results = model.evaluate(test_data, test_labels)
print(results)

# here we will create a graph for accuracy and the total loss over time
history_dict = history.history
history_dict.keys()
# the accuracy
acc = history_dict['acc']
# the value's accuracy
val_acc = history_dict['val_acc']
# the loss time
loss = history_dict['loss']
# the value's loss time
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# the bo stands for a blue dot
plt.plot(epochs, loss, 'blue dot', label='Training Loss')
# b stands for a solid blue line
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

