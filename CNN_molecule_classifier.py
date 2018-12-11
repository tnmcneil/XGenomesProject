from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout, Activation, Dense
from keras.optimizers import SGD, Adam
import glob, os
import random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. using non interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

CWD = "/usr4/cs542/tnmcneil"


def build_model():
    classifier = Sequential()

    # layer 1
    classifier.add(Conv2D(32,(3,3), input_shape=(960, 139, 2)))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    # layer 2
    classifier.add(Conv2D(64,(3,3)))
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    # flatten
    classifier.add(Flatten())

    # fully connected layer
    classifier.add(Dense(64))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.6))
    classifier.add(Dense(2))
    classifier.add(Activation('sigmoid'))

    #compile model

    classifier.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.99999, epsilon=1e-08, decay=0.0),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    # print summary of model
    classifier.summary()

    return classifier


# image processing
def process_data():

    cwd = CWD + "/column_data"

    # read all data

    lambda_dir1 = cwd + "/lambda_1"
    os.chdir(lambda_dir1)
    lambda_data1 = glob.glob("*.jpg")
    lambda_data1 = [lambda_dir1 + "/" + data for data in lambda_data1]
    
    lambda_dir2 = cwd + "/lambda_2"
    os.chdir(lambda_dir2)
    lambda_data2 = glob.glob("*.jpg")
    lambda_data2 = [lambda_dir2 + "/" + data for data in lambda_data2]

    lambda_dir3 = cwd + "/lambda_3"
    os.chdir(lambda_dir3)
    lambda_data3 = glob.glob("*.jpg")
    lambda_data3 = [lambda_dir3 + "/" + data for data in lambda_data3]

    lambda_dir4 = cwd + "/lambda_4"
    os.chdir(lambda_dir4)
    lambda_data4 = glob.glob("*.jpg")
    lambda_data4 = [lambda_dir4 + "/" + data for data in lambda_data4]

    lambda_data = lambda_data1 + lambda_data2 + lambda_data3 + lambda_data4

    t7_dir1 = cwd + "/t7_1_alligned"
    os.chdir(t7_dir1)
    t7_data1 = glob.glob("*.jpg")
    t7_data1 = [t7_dir1 + "/" + data for data in t7_data1]
    
    t7_dir2 = cwd + "/t7_2_alligned"
    os.chdir(t7_dir2)
    t7_data2 = glob.glob("*.jpg")
    t7_data2 = [t7_dir2 + "/" + data for data in t7_data2]
    
    t7_dir3 = cwd + "/t7_3_alligned"
    os.chdir(t7_dir3)
    t7_data3 = glob.glob("*.jpg")
    t7_data3 = [t7_dir3 + "/" + data for data in t7_data3]
    
    t7_dir4 = cwd + "/t7_4_alligned"
    os.chdir(t7_dir4)
    t7_data4 = glob.glob("*.jpg")
    t7_data4 = [t7_dir4 + "/" + data for data in t7_data4]
    
    t7_data = t7_data1 + t7_data2 + t7_data3 + t7_data4

    # train / test / validation data split

    random.shuffle(lambda_data)
    random.shuffle(t7_data)

    lambda_len = len(lambda_data)
    t_len = len(t7_data)

    l_train_cutoff = int(0.7 * lambda_len)
    l_test_cutoff = int(0.2 * lambda_len) + l_train_cutoff
    l_val_cutoff = int(0.1 * lambda_len) + l_test_cutoff

    t_train_cutoff = int(0.7 * t_len)
    t_test_cutoff = int(0.2 * t_len) + t_train_cutoff
    t_val_cutoff = int(0.1 * t_len) + t_test_cutoff

    lambda_train = lambda_data[:l_train_cutoff]
    lambda_test = lambda_data[l_train_cutoff:l_test_cutoff]
    lambda_val = lambda_data[l_test_cutoff:]

    t7_train = t7_data[:t_train_cutoff]
    t7_test = t7_data[t_train_cutoff:t_test_cutoff]
    t7_val = t7_data[t_test_cutoff:]

    # label data
    # [1.0, 0.0] = lambda; [0.0, 1.0] = t7

    train_files = lambda_train + t7_train
    random.shuffle(train_files)
    y_train = [None] * (len(train_files))
    i = 0
    for i in range(len(train_files)):
        if 'lambda' in train_files[i]:
            y_train[i] = [1.0, 0.0]
        else:
            y_train[i] = [0.0, 1.0]

    y_train = np.array(y_train)

    test_files = lambda_test + t7_test
    random.shuffle(test_files)
    y_test = [None] * (len(test_files))
    j = 0
    for j in range(len(test_files)):
        if 'lambda' in test_files[j]:
            y_test[j] = [1.0, 0.0]
        else:
            y_test[j] = [0.0, 1.0]

    y_test = np.array(y_test)

    val_files = lambda_val + t7_val
    random.shuffle(val_files)
    y_val = [None] * (len(val_files))
    k = 0
    for k in range(len(val_files)):
        if 'lambda' in val_files[k]:
            y_val[k] = [1.0, 0.0]
        else:
            y_val[k] = [0.0, 1.0]

    y_val = np.array(y_val)

    # convert images to numpy arrays

    training_set = np.ndarray(shape=(len(train_files), 960, 139, 2), dtype=np.float32)
    i = 0
    for _file in train_files:
        img = load_img(_file).convert('LA')
        x = img_to_array(img)
        result = np.zeros((960, 139, 2))
        result[:x.shape[0],:x.shape[1]] = x
        training_set[i] = result
        i += 1

    testing_set = np.ndarray(shape=(len(test_files), 960, 139, 2), dtype=np.float32)
    j = 0
    for _file in test_files:
        img = load_img(_file).convert('LA')
        x = img_to_array(img)
        result = np.zeros((960, 139, 2))
        result[:x.shape[0], :x.shape[1]] = x
        testing_set[j] = result
        j += 1

    val_set = np.ndarray(shape=(len(val_files), 960, 139, 2), dtype=np.float32)
    k = 0
    for _file in val_files:
        img = load_img(_file).convert('LA')
        x = img_to_array(img)
        result = np.zeros((960, 139, 2))
        result[:x.shape[0], :x.shape[1]] = x
        val_set[k] = result
        k += 1

    training_set = training_set.astype("float32")/255.0
    testing_set = testing_set.astype("float32")/255.0
    val_set = val_set.astype("float32")/255.0
    
    return training_set, y_train, testing_set, y_test, val_set, y_val


def calc_precision_recall(prediction, real):
    len_test = len(prediction)
    incorrect = 0
    a = 0
    for a in range(len_test):
        if not ((prediction[a] == real[a]).all()):
            incorrect += 1
    proportion_correct = 1 - (incorrect/len_test)

    a = 0
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for a in range(len_test):
        if (real[a] == [1.0, 0.0]).all() and (prediction[a] == [1.0,0.0]).all():
            tp += 1
        elif (real[a] == [1.0, 0.0]).all() and (prediction[a] == [0.0, 1.0]).all():
            fn += 1
        elif (real[a] == [0.0, 1.0]).all() and (prediction[a] == [0.0, 1.0]).all():
            tn += 1
        elif (real[a] == [0.0, 1.0]).all() and (prediction[a] == [1.0, 0.0]).all():
            fp += 1

    print("true positives = ", tp)
    print("false negatives = ", fn)
    print("true negatives = ", tn)
    print("false positives = ", fp)
    print('incorrect:', incorrect)
    print('proportion correct:', proportion_correct)
    precision_calc = tp / (tp + fp)
    recall_calc = tp / (tp + fn)
    print("my precision: ", precision_calc)
    print("my recall: ", recall_calc)

# Model Training Hyper parameters

EPOCHS = 100
BATCH_SIZE = 32

# create instance of model

classifier = build_model()

training_set, y_train, testing_set, y_test, val_set, y_val = process_data()

hist = classifier.fit(x=training_set, y=y_train, epochs=EPOCHS, verbose=2,
                      batch_size=BATCH_SIZE, validation_data=(val_set, y_val))

score = classifier.evaluate(testing_set, y_test, verbose=0)

hypothesis = classifier.predict(testing_set)

label = np.empty((len(testing_set), 2))

# summarize history for loss

plt.figure(1)
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper right")
plt.savefig(CWD + "/loss_history.png", bbox_inches='tight')

# summarize history for accuracy
plt.figure(2)
plt.plot(hist.history["acc"])
plt.plot(hist.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="lower right")
plt.savefig(CWD + "/acc_history.png", bbox_inches='tight')

i=0
for i in range(len(testing_set)):
    if hypothesis[i][0] > hypothesis[i][1]:
        label[i] = [1.0, 0.0]
    else:
        label[i] = [0.0, 1.0]

calc_precision_recall(label, y_test)