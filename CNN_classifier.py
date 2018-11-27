from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout, Activation, Dense
import glob, os
import random
import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display
from IPython.display import Image as _Imgdis
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import functools
from keras import backend as K
import tensorflow as tf

classifier = Sequential()

# layer 1
classifier.add(Conv2D(32,(50,10), input_shape=(512, 512, 2)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# layer 2
classifier.add(Conv2D(64,(10,5)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# layer 3
#classifier.add(Conv2D(64, (3,3)))
#classifier.add(Activation('relu'))
#classifier.add(MaxPooling2D(pool_size=(2,2)))

# flatten
classifier.add(Flatten())

classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

#classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# image processing

cwd = os.getcwd()

lambda_dir1 = cwd + "/data/lambda_1"
os.chdir(lambda_dir1)
lambda_data1 = glob.glob("*.jpg")
lambda_data1 = [lambda_dir1 + "/" + data for data in lambda_data1]

lambda_dir2 = cwd + "/data/lambda_2"
os.chdir(lambda_dir2)
lambda_data2 = glob.glob("*.jpg")
lambda_data2 = [lambda_dir2 + "/" + data for data in lambda_data2]

lambda_dir3 = cwd + "/data/lambda_3"
os.chdir(lambda_dir3)
lambda_data3 = glob.glob("*.jpg")
lambda_data3 = [lambda_dir3 + "/" + data for data in lambda_data3]

lambda_data = lambda_data1 + lambda_data2 + lambda_data3

t7_dir1 = cwd + "/data/T7_1"
os.chdir(t7_dir1)
t7_data1 = glob.glob("*.jpg")
t7_data1 = [t7_dir1 + "/" + data for data in t7_data1]

t7_dir2 = cwd + "/data/T7_2"
os.chdir(t7_dir2)
t7_data2 = glob.glob("*.jpg")
t7_data2 = [t7_dir2 + "/" + data for data in t7_data2]

t7_dir3 = cwd + "/data/T7_3"
os.chdir(t7_dir3)
t7_data3 = glob.glob("*.jpg")
t7_data3 = [t7_dir3 + "/" + data for data in t7_data3]

t7_data = t7_data1 + t7_data2 + t7_data3

random.shuffle(lambda_data)
random.shuffle(t7_data)

lambda_len = len(lambda_data)
t_len = len(t7_data)

l_train_cutoff = int((0.7)*(lambda_len))
l_test_cutoff = int((0.2)*(lambda_len)) + l_train_cutoff
l_val_cutoff = int((0.1)*(lambda_len)) + l_test_cutoff

t_train_cutoff = int((0.7)*(t_len))
t_test_cutoff = int((0.2)*(t_len)) + t_train_cutoff
t_val_cutoff = int((0.1)*(t_len)) + t_test_cutoff

lambda_train = lambda_data[:l_train_cutoff]
lambda_test = lambda_data[l_train_cutoff:l_test_cutoff]
lambda_val = lambda_data[l_test_cutoff:]

t7_train = t7_data[:t_train_cutoff]
t7_test = t7_data[t_train_cutoff:t_test_cutoff]
t7_val = t7_data[t_test_cutoff:]

#0.0 = lambda; 1.0 = t7

train_files = lambda_train + t7_train
y_train = [0.0 for i in range(len(lambda_train))] + [1.0 for j in range(len(t7_train))]

test_files = lambda_test + t7_test
y_test = [0.0 for i in range(len(lambda_test))] + [1.0 for j in range(len(t7_test))]

val_files = lambda_val + t7_val
y_val = [0.0 for i in range(len(lambda_val))] + [1.0 for j in range(len(t7_val))]

training_set = np.ndarray(shape=(len(train_files), 512, 512, 2), dtype=np.float32)
i = 0
for _file in train_files:
    img = load_img(_file).convert('LA')
    x = img_to_array(img)
    training_set[i] = x
    i += 1

testing_set = np.ndarray(shape=(len(test_files), 512, 512, 2), dtype=np.float32)
j = 0
for _file in test_files:
    img = load_img(_file).convert('LA')
    x = img_to_array(img)
    testing_set[j] = x
    j += 1

val_set = np.ndarray(shape=(len(val_files), 512, 512, 2), dtype=np.float32)
k = 0
for _file in val_files:
    img = load_img(_file).convert('LA')
    x = img_to_array(img)
    val_set[k] = x
    k += 1

#training_set = training_set.astype("float32")/255
#testing_set = testing_set.astype("float32")/255
#val_set = val_set.astype("float32")/255

def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
             value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def auc_pr(y_true, y_pred, curve='PR'):
    return tf.metrics.auc(y_true, y_pred, curve=curve)

precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy', precision, recall, auc_pr])

classifier.fit(x=training_set, y = y_train, epochs = 25, validation_data = (val_set, y_val))

score = classifier.evaluate(testing_set, y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
print('precision:', score[2])
print('recall', score[3])
print('auc', score[4])

'''
print("Working with {0} lambda images".format(len(lambda_data)))

print("Image Examples: ")
for i in range(40,42):
    print(lambda_data[i])
    display(_Imgdis(filename= cwd + lambda_data[i]))

print("Working with {0} t7 images".format(len(lambda_data)))

print("Image Examples: ")
for i in range(40,42):
    print(t7_data[i])
    display(_Imgdis(filename= cwd + t7_data[i]))
'''
