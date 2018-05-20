#!/usr/bin/python3
import time
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score


def show_image(pixels, labels, ind, shape):

    pixels = pixels[ind].reshape(shape)
    
    plt.imshow(pixels, cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.title(int(labels[ind]))
    plt.axis("off")
    plt.show()


# This function is from Hands-On Machine Learning with Scikit-Learn and Tensorflow
def neuron_layer(X, n_neurons, name, activation = None):

    with tf.name_scope(name):

        n_inputs = int(X.get_shape()[1])

        stddev = 2/np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)

        W = tf.Variable(init, name = "kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name = "bias")

        Z = tf.matmul(X, W) + b

        if activation is not None:
            return activation(Z)

        else:
            return Z
        
        
if __name__ == "__main__":

    # Load the data
    mnist = sio.loadmat('/home/ivy/scikit_learn_data/mldata/mnist-original', squeeze_me = True)

    print("Loaded data\n\n")
    
    X, Y = mnist["data"].T, mnist["label"].T

    print("X has dimensions", X.shape)
    print("Y has dimensions", Y.shape)
    print("\n\n")

    m = X.shape[0]
    
    #Y_mod = label_binarize(Y, classes = range(10))

    n_classes = 10
    
    print("n_classes:", n_classes)
    print()
    
    ## Split data into train, test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 10000, random_state = 8012, shuffle = True)

    ## Further split train into validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 5000, random_state = 9012, shuffle = True)

    m_train = X_train.shape[0]
    m_val = X_val.shape[0]
    m_test = X_test.shape[0]
    
    print("m_train", m_train)
    print("m_val", m_val)
    print("m_test", m_test)
    

    ## For debugging only
    # X_train = X_train[0:200]
    # Y_train = Y_train[0:200]


    n_inputs = X.shape[1]
    n_hidden1 = 300
    n_outputs = 10

    #for activ in [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, tf.nn.leaky_relu, tf.nn.elu, tf.nn.relu6]:

    #tf.reset_default_graph()
        
    X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
    Y = tf.placeholder(tf.int64, shape = (None), name = "Y")
    
    
    with tf.name_scope("NN_1L"):
        
        hidden1 = tf.layers.dense(X, n_hidden1, name = "hidden1",
                                      activation = tf.nn.relu)

        logits = tf.layers.dense(hidden1, n_outputs, name = "outputs")
        

    with tf.name_scope("loss"):

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = Y,
                                                                       logits = logits)
                
        loss = tf.reduce_mean(cross_entropy, name = "loss")

        learning_rate = 0.01

    with tf.name_scope("train"):
                
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              momentum = 0.9,
                                              decay = 0.9,
                                              epsilon = 1e-10)

        training_op = optimizer.minimize(loss)
                
    with tf.name_scope("eval"):

        #correct = tf.nn.in_top_k(logits, Y, 1)
        #accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            
        prediction = tf.argmax(logits, axis = 1)
        precision, precision_op = tf.metrics.precision(Y, tf.argmax(logits, axis = 1))
        recall, recall_op = tf.metrics.recall(Y, prediction)
            

            

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    saver = tf.train.Saver

    n_epochs = 200
    #batch_size = 10

    with tf.Session() as sess:

        print()
        print("Starting to train with ReLu activation and Gradient Descent")

            
        start_time = time.time()
            
        init.run()
        init_local.run()
            
        for epoch in range(n_epochs):

            sess.run(training_op, feed_dict = {X: X_train, Y: Y_train})
            prec_train = sess.run(precision_op, feed_dict = {X: X_train, Y: Y_train})
            recall_train = sess.run(recall_op, feed_dict = {X: X_train, Y: Y_train})

            prec_val = sess.run(precision_op, feed_dict = {X: X_val, Y: Y_val})
            recall_val = sess.run(recall_op, feed_dict = {X: X_val, Y: Y_val})
                
            print("Epoch ", epoch, "Train precision:", prec_train, "Train recall:", recall_train)
            print("Epoch:", epoch, "Val precision:", prec_val, "Val recall:", recall_val)
            print()
                

        print("Time elapsed", (time.time() - start_time)/60, "minutes")
        print()
