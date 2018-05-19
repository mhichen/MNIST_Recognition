#!/usr/bin/python3
import time
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

def show_image(pixels, labels, ind, shape):

    pixels = pixels[ind].reshape(shape)
    
    plt.imshow(pixels, cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.title(int(labels[ind]))
    plt.axis("off")
    plt.show()

        
if __name__ == "__main__":


    # Load the data
    mnist = sio.loadmat('/home/ivy/scikit_learn_data/mldata/mnist-original', squeeze_me = True)

    print("Loaded data\n\n")
    
    X, Y = mnist["data"].T, mnist["label"].T

    print("X has dimensions", X.shape)
    print("Y has dimensions", Y.shape)
    print("\n\n")

    Y_mod = label_binarize(Y, classes = range(10))

    n_classes = Y_mod.shape[1]

    print("n_classes:", n_classes)
    print()
    
    # Check a few images
    # show_image(X, Y, 56000, (28, 28))

    ## Split data into train, test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_mod, test_size = 0.2, random_state = 8012, shuffle = True)

    ## For debugging only
    # X_train = X_train[1:2000]
    # Y_train = Y_train[1:2000]
    
    ##***********************************************##
    ## Try different K's:                            ##
    ## K - Nearest Neighbors Classification          ##
    ##***********************************************##

    for K in range(1, 11):

        print("Fitting for K = ", K)
        
        start_time = time.time()
        
        neigh = KNeighborsClassifier(n_neighbors = K, weights = 'uniform',
                                 algorithm = 'auto', leaf_size = 30,
                                 p = 2, metric = 'minkowski',
                                     metric_params = None, n_jobs = 2)



        scores = cross_val_score(neigh, X_train, Y_train, scoring = 'average_precision', cv = 3)

        print("Average precision scores across segments")
        print(scores)

        print("Time elapsed", (time.time() - start_time)/60, "minutes")
        print()


    start_time = time.time()
    
    K = 4
    
    neigh = KNeighborsClassifier(n_neighbors = K, weights = 'uniform',
                                 algorithm = 'auto', leaf_size = 30,
                                 p = 2, metric = 'minkowski',
                                 metric_params = None, n_jobs = 2)


        
    neigh.fit(X_train, Y_train)
    
    # Save model
    filename = 'kNN_K' + str(K) + 'final.sav'
    pickle.dump(neigh, open(filename, 'wb'))
    
    Y_train_predicted = neigh.predict(X_train)
    
    report = classification_report(Y_train, Y_train_predicted, target_names = [str(x) for x in range(n_classes)])
    
    print("Classification report on training data")
    print(report)
    
    print("Time elapsed", (time.time() - start_time)/60, "minutes")
    print()

    # Classification report on training data
    #              precision    recall  f1-score   support
    
    #           0       0.99      0.99      0.99      5508
    #           1       0.98      0.99      0.99      6295
    #           2       1.00      0.96      0.98      5548
    #           3       0.99      0.97      0.98      5690
    #           4       1.00      0.97      0.98      5495
    #           5       1.00      0.96      0.98      5063
    #           6       0.99      0.99      0.99      5504
    #           7       0.99      0.98      0.98      5867
    #           8       1.00      0.93      0.96      5444
    #           9       0.99      0.96      0.98      5586
    
    # avg / total       0.99      0.97      0.98     56000
    
    # Time elapsed 58.82667687336604 minutes

    start_time = time.time()

    model = pickle.load(open("kNN_K4final.sav", 'rb'))
    Y_test_predicted = model.predict(X_test)

    report = classification_report(Y_test, Y_test_predicted, target_names = [str(x) for x in range(n_classes)])
    
    print("Classification report on test data")
    print(report)
    
    print("Time elapsed", (time.time() - start_time)/60, "minutes")
    print()

    # Classification report on test data
    #              precision    recall  f1-score   support
    
    #           0       0.99      0.99      0.99      1395
    #           1       0.98      0.99      0.99      1582
    #           2       0.99      0.96      0.97      1442
    #           3       0.99      0.95      0.97      1451
    #           4       0.99      0.96      0.97      1329
    #           5       0.98      0.95      0.97      1250
    #           6       0.99      0.98      0.99      1372
    #           7       0.98      0.97      0.97      1426
    #           8       0.99      0.91      0.95      1381
    #           9       0.97      0.95      0.96      1372
    
    # avg / total       0.99      0.96      0.97     14000
    
    # Time elapsed 14.392941125233968 minutes
