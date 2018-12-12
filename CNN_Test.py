import tensorflow as tf
import tempfile
import cv2
import random
import numpy as np
import os

# Copyright 2018 Yang Kaiyu kyyang@bu.edu
class loadimage():
    """This class is to handle the image loading process"""
    def __init__(self, datalist):
        """Init class, datalist is image file's path with label"""
        self._data = datalist.copy()
        self._rpoint = 0
        self._lenth = len(datalist)
    
    def next_batch(self, size=-1):
        """load a certain number of image files and assemble as training data"""
        thisbatch = []
        if size == -1:
            size = self._lenth
        
        while self._rpoint+size >= self._lenth: # handle large batch size
            thisbatch = thisbatch + self._data[self._rpoint:]
            random.shuffle(self._data)
            size -= len(self._data[self._rpoint:])
            self._rpoint = 0
        thisbatch = thisbatch + self._data[self._rpoint:self._rpoint+size]
        self._rpoint += size
        
        imglist = []
        taglist = []
        for sample in thisbatch: # load image from file
            imglist.append(np.expand_dims(np.float32(cv2.imread(sample[0],0))/255,3))
            taglist.append(np.array([sample[1],1-sample[1]]))
        return (np.array(imglist),np.array(taglist))
    
    def reset(self):
        """reset read pointer and shuffle the data"""
        self._rpoint = 0
        random.shuffle(self._data)
        self._lenth = len(datalist)

####################################################
################Adjustable Features#################
set_name = "Set123_4"  # data set folder for testing   ./set_name
Model_name = "Set_regen/Set_regen"  # which model to test   ./Model_name
ClassA_test_name = "Lambda_test"  # ClassA folder in data set folder   ./set_name/ClassA_test_name
ClassB_test_name = "T7_test"  # ClassB folder in data set folder   ./set_name/ClassB_test_name
batch_size = 50
################Adjustable Features#################
####################################################

#####Prepare Data#####
ClassA_test = [["./"+set_name+"/"+ClassA_test_name+"/"+aname,0] for aname in os.listdir("./"+set_name+"/"+ClassA_test_name+"/")]
ClassB_test = [["./"+set_name+"/"+ClassB_test_name+"/"+aname,1] for aname in os.listdir("./"+set_name+"/"+ClassB_test_name+"/")]
print(ClassA_test_name, "has",len(ClassA_test),"testing samples")
print(ClassB_test_name, "has",len(ClassB_test),"testing samples")

print("Generate testing set")
Test_list = ClassA_test + ClassB_test
random.shuffle(Test_list)
print(len(Test_list),"testing samples...Done")

print("Prepare image reader")
test_set = loadimage(Test_list)
#######Done#######

#####Load Meta Graph#####
saver = tf.train.import_meta_graph("./"+Model_name+".meta")
accuracy = tf.get_collection("accuracy")[0]
x = tf.get_collection("x")[0]
y_ = tf.get_collection("y_")[0]
recall = tf.get_collection("recall")[0]
precision = tf.get_collection("precision")[0]
train_step = tf.get_collection("train_step")[0]
#######Done#######


with tf.Session() as sess:
    saver.restore(sess,"./"+Model_name) # Restore graph
    accuracy_all = []
    recall_all = []
    precision_all = []
    for i in range(0,len(Test_list),batch_size):
        print("Progress",str(int(i/len(Test_list)*100))+"%", end="\r")
        test = test_set.next_batch(batch_size)
        test_recall, test_precision, test_accuracy = sess.run([recall, precision, accuracy],{x: test[0], y_: test[1]})
        accuracy_all.append(test_accuracy)
        recall_all.append(test_recall)
        precision_all.append(test_precision)
        
    print("")
    print("test accuracy", sum(accuracy_all) / len(accuracy_all))
    print("test recall", sum(recall_all) / len(recall_all))
    print("test precision", sum(precision_all) / len(precision_all))