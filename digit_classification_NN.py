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

from datetime import datetime

def show_image(pixels, labels, ind, shape):

    pixels = pixels[ind].reshape(shape)
    
    plt.imshow(pixels, cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.title(int(labels[ind]))
    plt.axis("off")
    plt.show()
    
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  
    indices = np.random.randint(m, size=batch_size)  
    X_batch = scaled_housing_data_plus_bias[indices] 
    y_batch = housing.target.reshape(-1, 1)[indices] 
    return X_batch, y_batch
    
if __name__ == "__main__":

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    
    # Load the data
    mnist = sio.loadmat('/home/ivy/scikit_learn_data/mldata/mnist-original', squeeze_me = True)

    print("Loaded data\n\n")
    
    X, Y = mnist["data"].T, mnist["label"].T

    print("X has dimensions", X.shape)
    print("Y has dimensions", Y.shape)
    print("\n\n")

    print(X[0].shape)

    X = np.reshape(X, (-1, 28, 28, 1))

    # plt.imshow(test[8000, :, :, 0], cmap = "gray")
    # plt.show()
    
    X = np.pad(X, ((0, 0), (2, 2), (2, 2), (0,0)), 'constant')
    
    print("test has dimensions", X.shape)
    
    
    #Y_mod = label_binarize(Y, classes = range(10))

    n_classes = 10
    
    print("n_classes:", n_classes)
    print()
    
    ## Split data into train, test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 10000, random_state = 8012, shuffle = True)

    ## Further split train into validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 5000, random_state = 9012, shuffle = True)


    ## For debugging only
    # X_train = X_train[0:200, :, :, :]
    # Y_train = Y_train[0:200]

    m_train = X_train.shape[0]
    m_val = X_val.shape[0]
    m_test = X_test.shape[0]

    n_inputs = X.shape[0]
    n_outputs = 10
    n_epochs = 200
    batch_size = 10
    
    learning_rate = 0.01
        
    print("m_train", m_train)
    print("m_val", m_val)
    print("m_test", m_test)
    

    X = tf.placeholder(tf.float32, shape = (None, 32, 32, 1), name = "X")
    Y = tf.placeholder(tf.int64, shape = (None), name = "Y")

    train_writer = tf.summary.FileWriter(logdir + '/train', tf.get_default_graph())
    val_writer = tf.summary.FileWriter(logdir + '/val', tf.get_default_graph())

    with tf.name_scope("LeNet-5"):

        # C1 - Convolution
        C1 = tf.layers.conv2d(X, filters = 6, kernel_size = [5, 5], strides = (1, 1),
                                   padding = 'valid', activation = tf.nn.tanh)

        # S2 - Average Pooling
        S2 = tf.layers.average_pooling2d(C1, pool_size = [2, 2], strides = (2, 2),
                                         padding = 'valid')

        # C3 - Convolution
        C3 = tf.layers.conv2d(S2, filters = 16, kernel_size = [5, 5], strides = (1, 1),
                                   padding = 'valid', activation = tf.nn.tanh)

        # S4 - Average Pooling
        S4 = tf.layers.average_pooling2d(C3, pool_size = [2, 2], strides = (2, 2),
                                         padding = 'valid')

        # C5 - Convolution
        C5 = tf.layers.conv2d(S4, filters = 16, kernel_size = [5, 5], strides = (1, 1),
                                   padding = 'valid', activation = tf.nn.tanh)
        
        F6 = tf.layers.dense(C5, units = 120, activation = tf.nn.tanh)

        F7 = tf.layers.dense(F6, units = 84, activation = tf.nn.tanh)

        logits = tf.layers.dense(F7, units = n_outputs)

        logits = tf.layers.flatten(logits)

        tf.summary.histogram('C1', C1)

        tf.summary.histogram('S2', S2)

    with tf.name_scope("loss"):

        loss = tf.losses.sparse_softmax_cross_entropy(labels = Y, logits = logits)

    with tf.name_scope("train"):

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        
    with tf.name_scope("eval"):

        correct = tf.nn.in_top_k(logits, Y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            
        prediction = tf.argmax(logits, axis = 1)
        precision, precision_op = tf.metrics.precision(Y, tf.argmax(logits, axis = 1))
        recall, recall_op = tf.metrics.recall(Y, prediction)

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)

        write_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    
    with tf.Session() as sess:

        start_time = time.time()
            
        init.run()
        init_local.run()

        n_batches = int(m_train/batch_size)

        print("n_batches", n_batches)
        print()
        
        for epoch in range(n_epochs):

            for b in range(n_batches):

                # print("batch b", b, b*batch_size, (b + 1)*batch_size)

                if (b - 1) != n_batches:
                    X_batch = X_train[b*batch_size:(b + 1)*batch_size, :, :, :]
                    Y_batch = Y_train[b*batch_size:(b + 1)*batch_size]

                else:
                    X_batch = X_train[b*batch_size:, :, :, :]
                    Y_batch = Y_train[b*batch_size:]

                # print("X_batch", X_batch.shape)
                # print("Y_batch", Y_batch.shape)
                
                sess.run(train_op, feed_dict = {X: X_batch, Y: Y_batch})

                _, prec_train, recall_train, accuracy_train = sess.run([train_op, precision_op, recall_op, accuracy], feed_dict = {X: X_batch, Y: Y_batch})
                #recall_train = sess.run(recall_op, feed_dict = {X: X_batch, Y: Y_batch})
            
                train = write_op.eval(feed_dict = {X: X_batch, Y: Y_batch})
                train_writer.add_summary(train, epoch)
                
            prec_val, recall_val, accuracy_val = sess.run([precision_op, recall_op, accuracy], feed_dict = {X: X_val, Y: Y_val})
            #recall_val = sess.run(recall_op, feed_dict = {X: X_val, Y: Y_val})

            validation = write_op.eval(feed_dict = {X: X_val, Y: Y_val})
            val_writer.add_summary(validation, epoch)
            
            print("Epoch:", epoch, "Train precision:", prec_train, "Train recall:", recall_train, "Train accuracy:", accuracy_train)
            print("Epoch:", epoch, "Val precision:", prec_val, "Val recall:", recall_val, "Val accuracy:", accuracy_val)
            print()

    train_writer.close()
    val_writer.close()
    
    print("Time elapsed", (time.time() - start_time)/60, "minutes")
    # print()
