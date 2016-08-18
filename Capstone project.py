
# coding: utf-8

# In[1]:

import tensorflow as tf
import scipy.io as scp
import numpy as np
import random
from scipy import misc
import matplotlib.pyplot as plt
import datetime

#To plot the graphs in jupyter
get_ipython().magic(u'matplotlib inline')


# In[2]:


# Load SVHN data in variables
trainData = scp.loadmat('../data/train_32x32.mat')
testData=scp.loadmat('../data/test_32x32.mat')


# In[3]:

# shape of data for images and labels
print trainData['X'].shape
print trainData['y'].shape
print testData['X'].shape
print testData['y'].shape


# In[4]:

# normalizing the data in range of -1 to 1. as the images are in format of 32*32 in 3 channels for RGB i will use a factor of 128 to divide datasets and subtract 1 to make it in a range of -1 to 1

trainDataX = trainData['X'].astype('float32') / 128.0 - 1                                                                                                                     
testDataX = testData['X'].astype('float32') / 128.0 - 1 

trainDataY=trainData['y']
testDataY=testData['y']


# In[5]:

# function for one hot encoding so that machine can easily differntiate between differnet classes and increase accuracy .in this we have to change label for image representing 0 as by default it's label is 10 
def OnehotEndoding(Y):
    Ytr=[]
    for el in Y:
        temp=np.zeros(10)
        if el==10:
            temp[0]=1
        else:
            temp[el] = 1
        Ytr.append(temp)

    return np.asarray(Ytr)


# In[6]:

trainDataY = OnehotEndoding(trainDataY)
testDataY = OnehotEndoding(testDataY)


# In[7]:

# I am changing the shape of training as well as testing data from (32,32,3,73257) to (73252,32,32,3) so that this shape matches the one of our label that is (73257,1)

def transposeArray(data):
    xtrain = []
    trainLen = data.shape[3]
    
  
    for x in xrange(trainLen):
        xtrain.append(data[:,:,:,x])
      

    xtrain = np.asarray(xtrain)
    return xtrain

trainDataX = transposeArray(trainDataX)
testDataX = transposeArray(testDataX)


# In[8]:

print trainDataX.shape
print testDataX.shape


# In[9]:


# Function to initialize weights. It is good to initialize weights with small amount of noise for symmetry breaking, and to prevent 0 gradients. 
def weight_variable(shape):
    """Args:
            shape: a list of 4 integer [patch size,patch size,channels,depth]"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


#Function to initialize bias
def bias_variable(shape):
    """Args:
            shape: a list containing depth"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution Function. In this i am using stride of one and are zero padded so that the output is the same size as the input
def conv2d(x, W):
    """Args:
            input: matrix of input to the convolution layer 
            weights: weights Matrix"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Pooling Function. In this is used for 2X2 max_pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# In[10]:

sess = tf.InteractiveSession()

#x,_y are placeholders for values that we'll input when we ask Tensorflow to run computation


x = tf.placeholder(tf.float32, shape=[None, 32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])



# First convolution layer.
# It will consist of convolution, followed by max pooling. 
# The convolutional will compute 32 features for each 5x5 patch. 
# Its weight tensor will have a shape of [5, 5, 3, 32].
# The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. 
# We will also have a bias vector with a component for each output channel. 

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,32,32,3])

# Applying convolution between x_image and weight tensor and then apllying max pooling.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Layer of Convolution similar to first one to make network deep. This layer will have 64 features for each 5X5 patch
W_conv2=weight_variable([5, 5, 32, 64])
b_conv2=bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
# Now that the image size has been reduced to 8x8, we add a fully-connected layer with 1024 neurons
# to allow processing on the entire image.
# We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.


W_fc1=weight_variable([8*8*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout 
# To reduce overfitting, we will apply dropout before the readout layer.

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



# Readout Layer
# A softmax layer.

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv=tf.nn.softmax(logits)


# In[11]:


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))



# Optimizer.
# Using more sophosticted Adams Optimizer for contrilling learning rate than steepest gradient descent optimizer

with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# correct Predictions

with tf.name_scope('correct_predictions'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# accuracy 
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# Create a summary to monitor cost tensor
tf.scalar_summary("cross_entropy", cross_entropy)
# Create a summary to monitor accuracy tensor
tf.scalar_summary("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.merge_all_summaries()





# In[15]:

print  'started at : ', str(datetime.datetime.now())
with tf.Session() as sess:
    epoch=20000
    batch_size=64
    
    #train_writer = tf.train.SummaryWriter('/Users/prince/Desktop/Svhn/tensorflow',graph=tf.get_default_graph())

    train_writer = tf.train.SummaryWriter('/tmp/mnist_logs',
                                      sess.graph)
    
    sess.run(tf.initialize_all_variables())
    
    
    p = np.random.permutation(range(len(trainDataX)))
    trX, trY = trainDataX[p], trainDataY[p]
  
    start = 0
    end = 0  
    for step in range(epoch):
        start = end
        end = start + batch_size

        if start >= len(trainDataX):
            start = 0
            end = start + batch_size

        if end >= len(trainDataX):
            end = len(trainDataX) - 1
            
        if start == end:
            start = 0
            end = start + batch_size
        
        inX, outY = trX[start:end], trY[start:end]
        #_, summary=sess.run([train_step, merged_summaries], feed_dict={x: inX, y_: outY, keep_prob: 0.5})
        
        _, summary = sess.run([train_step, merged_summary_op], feed_dict= {x: inX, y_: outY, keep_prob:0.5})
        train_writer.add_summary(summary, step)
        #train_step.run(feed_dict={x: inX, y_: outY, keep_prob: 0.5})
        if step % 500 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: inX, y_: outY, keep_prob:1})
            print("step %d, training accuracy %g"%(step, train_accuracy))
            
        
        

    print("test accuracy %g"%accuracy.eval(feed_dict={
    x: testDataX, y_:testDataY , keep_prob: 1.0}))
    

print  'Ended at : ', str(datetime.datetime.now())

    

    
    
    
    


# In[ ]:

# Results on dual core mac with 8 gb ram

# For 10000 epochs and a batch size of 64 
# testing Accuracy 88.43%
# total time taken 59 min 


# For 20000 epochs and a batch size of 64
# Testing Accuracy 89.65%
# total time 45 min

#For 10000 epochs and a batch size of 16
#Testing Accuracy 85.66%
#Total time 27 min




