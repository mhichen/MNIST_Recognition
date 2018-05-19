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



        
    #neigh.fit(X_train, Y_train)
    #     # Save model
    #     filename = 'kNN_K' + str(K) + '.sav'
    #     pickle.dump(neigh, open(filename, 'wb'))
    
    #     Y_train_predicted = neigh.predict(X_train)

    #     report = classification_report(Y_train, Y_train_predicted, target_names = [str(x) for x in range(n_classes)])

    #     print("Classification report on training data")
    #     print(report)

    #     scores = cross_val_score(neigh, X_train, Y_train, cv = 5)
    #     print(scores)
        
    #     print("Time elapsed", (time.time() - start_time)/60, "minutes")
    #     print()

    # #model = pickle.load(open("kNN_K1.sav", 'rb'))
    # #Y_test_predicted = model.predict(X_test)
