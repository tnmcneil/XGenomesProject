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

classifier = Sequential()

# layer 1
classifier.add(Conv2D(32,(3,3), input_shape=(512, 512, 2)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# layer 2
classifier.add(Conv2D(32,(3,3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# layer 3
classifier.add(Conv2D(64, (3,3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# flatten
classifier.add(Flatten())

classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# image processing

cwd = os.getcwd()

lambda_dir = cwd + "/processed_data/lambda"
os.chdir(lambda_dir)
lambda_data = glob.glob("*.jpg")
lambda_data = ["/processed_data/lambda/" + data for data in lambda_data]

t7_dir = cwd + "/processed_data/t7"
os.chdir(t7_dir)
t7_data = glob.glob("*.jpg")
t7_data = ["/processed_data/t7/" + data for data in t7_data]

random.shuffle(lambda_data)
random.shuffle(t7_data)

lambda_train = lambda_data[:4000]
t7_train = t7_data[:4000]
lambda_test = lambda_data[4000:]
t7_test = t7_data[4000:]

train_files = lambda_train + t7_train
#y_train = ["lambda" for i in range(4000)] + ["t7" for j in range(4000)]
y_train = [0.0 for i in range(4000)] + [1.0 for j in range(4000)]

test_files = lambda_test + t7_test
#y_test = ["lambda" for i in range(1000)] + ["t7" for j in range(1000)]
y_test = [0.0 for i in range(1000)] + [1.0 for j in range(1000)]

training_set = np.ndarray(shape=(8000, 512, 512, 2), dtype=np.float32)
i = 0
for _file in train_files:
    img = load_img(cwd + _file).convert('LA')
    x = img_to_array(img)
    training_set[i] = x
    if i > 40 and i < 45:
        print(x.shape)
    i += 1

testing_set = np.ndarray(shape=(2000, 512, 512, 2), dtype=np.float32)
j = 0
for _file in test_files:
    img = load_img(cwd + _file).convert('LA')
    x = img_to_array(img)
    testing_set[j] = x
    if j > 40 and j < 45:
        print(x.shape)
    j += 1

# taken from old Keras documentation

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives/ (possible_positives + K.epsilon())
    return recall

classifier.fit(x=training_set, y = y_train, epochs = 25, validation_data = (testing_set, y_test))

score = classifier.evaluate(testing_set, y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

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
