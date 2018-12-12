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

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(input=x,filter=W,strides=[1, 1, 1, 1],padding="SAME")

def max_pool_def(x,size):
    """max_pool downsamples a feature map by a factor."""
    return tf.nn.max_pool(value=x,ksize=[1, size[0], size[1], 1],strides=[1, size[0], size[1], 1],padding="SAME")


####################################################
################Adjustable Features#################
set_name = "Set_fake"  # Input training data folder   ./set_name
Model_name = "Set_fake"  # Output model name  ./set_name/Model_name
ClassA_name = "Lambda_train"  # Input training data ClassA's folder  ./set_name/ClassA_name
ClassB_name = "T7_train"  # Input training data ClassB's folder  ./set_name/ClassB_name
ClassA_test_name = "Lambda_test"  # Input testing data ClassA's folder  ./set_name/ClassA_test_name
ClassB_test_name = "T7_test"  # Input testing data ClassB's folder  ./set_name/ClassA_test_name
train_epoch = 400
batch_size = 40
train_rate = 0.00001

Image_size = [640,640] # [height,width]

Conv_Kernal_1 = [50,10] # First convolution kernal
Conv_Feature_1 = 32 # First convolution feature number
pooling_size_1 = [16,16] # First downsampling factor

Conv_Kernal_2 = [10,5] # Second convolution kernal
Conv_Feature_2 = 32 # Second convolution feature number
pooling_size_2 = [8,8] # Second downsampling factor

Ful_Feature = 256 # Fully connected layer feature number

################Adjustable Features#################
####################################################

#####Prepare Data#####
ClassA_train = [["./"+set_name+"/"+ClassA_name+"/"+aname,0] for aname in os.listdir("./"+set_name+"/"+ClassA_name+"/")]
ClassB_train = [["./"+set_name+"/"+ClassB_name+"/"+aname,1] for aname in os.listdir("./"+set_name+"/"+ClassB_name+"/")]
ClassA_test = [["./"+set_name+"/"+ClassA_test_name+"/"+aname,0] for aname in os.listdir("./"+set_name+"/"+ClassA_test_name+"/")]
ClassB_test = [["./"+set_name+"/"+ClassB_test_name+"/"+aname,1] for aname in os.listdir("./"+set_name+"/"+ClassB_test_name+"/")]
print(ClassA_name, "has",len(ClassA_train),"training samples and",len(ClassA_test),"testing samples")
print(ClassB_name, "has",len(ClassB_train),"training samples and",len(ClassB_test),"testing samples")

print("Generate training set")
Train_list = ClassA_train + ClassB_train
random.shuffle(Train_list)
print(len(Train_list),"training samples...Done")
print("Generate testing set")
Test_list = ClassA_test + ClassB_test
random.shuffle(Test_list)
print(len(Test_list),"testing samples...Done")

print("Prepare image reader")
train_set = loadimage(Train_list)
test_set = loadimage(Test_list)
#######Done#######

print("Start building CNN graph")
print("Building input & output placeholder")
x = tf.placeholder(tf.float32, [None, Image_size[0],Image_size[1], 1])
y_ = tf.placeholder(tf.float32, [None, 2])

print("Building First convolution layer")
w_conv1 = weight_variable([Conv_Kernal_1[0], Conv_Kernal_1[1], 1, Conv_Feature_1])
b_conv1 = bias_variable([Conv_Feature_1])
h_conv1 = tf.nn.relu(conv2d(x,w_conv1) + b_conv1)

print("Building First pooling layer")
h_pool1 = max_pool_def(h_conv1,pooling_size_1)

print("Building Second convolution layer")
w_conv2 =  weight_variable([Conv_Kernal_2[0], Conv_Kernal_2[1], Conv_Feature_1, Conv_Feature_2])
b_conv2 = bias_variable([Conv_Feature_2])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)

print("Building Second pooling layer")
h_pool2 = max_pool_def(h_conv2,pooling_size_2)

print("Building First Fully connected layer")
h_pool2_size = Image_size[0] * Image_size[1]//(pooling_size_1[0] * pooling_size_1[1]) //(pooling_size_2[0]*pooling_size_2[1]) * Conv_Feature_2
w_fc1 = weight_variable([h_pool2_size, Ful_Feature])
b_fc1 = bias_variable([Ful_Feature])
h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_size])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

print("Building Second Fully connected layer")
w_fc2 = weight_variable([Ful_Feature, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1,w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(train_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
true_positives = tf.multiply(tf.argmax(y_conv,1),tf.argmax(y_,1))
precision = tf.reduce_sum(tf.cast(true_positives, "float"))/tf.reduce_sum(tf.cast(tf.argmax(y_conv,1), "float"))
recall = tf.reduce_sum(tf.cast(true_positives, "float"))/tf.reduce_sum(tf.cast(tf.argmax(y_,1), "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#####Save graph#####
tf.add_to_collection('x', x)
tf.add_to_collection('y_', y_)
tf.add_to_collection('y_conv', y_conv)
tf.add_to_collection('recall', recall)
tf.add_to_collection('precision', precision)
tf.add_to_collection('accuracy', accuracy)
tf.add_to_collection('train_step', train_step)
saver = tf.train.Saver()
saver.export_meta_graph("./"+set_name+"/"+Model_name+".meta")
#####Done#####

log_file = open("./"+set_name+"/log.csv",'w', buffering=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_epoch):
        batch = train_set.next_batch(batch_size)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            test = test_set.next_batch(batch_size)
            test_recall, test_precision, test_accuracy = sess.run([recall, precision, accuracy],{x: test[0], y_: test[1]})
            print('step %d, training accuracy %g, test accuracy %g, recall %g, precision %g' % (i, train_accuracy, test_accuracy, test_recall,test_precision))
            log_file.write('%d, %g, %g, %g, %g\n' % (i, train_accuracy, test_accuracy, test_recall, test_precision))
    saver.save(sess,"./"+set_name+"/"+Model_name)

log_file.close()