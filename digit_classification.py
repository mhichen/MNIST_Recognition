import time
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

def show_image(pixels, labels, ind, shape):

    pixels = pixels[ind].reshape(shape)
    
    plt.imshow(pixels, cmap = matplotlib.cm.binary, interpolation = "nearest")
    plt.title(int(labels[ind]))
    plt.axis("off")
    plt.show()

        
if __name__ == "__main__":


    start_time = time.time()
    
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
    #show_image(X, Y, 56000, (28, 28))

    ## Split data into train, test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_mod, test_size = 0.2, random_state = 8012, shuffle = True)

    ## For debugging only
    # X_train = X_train[1:200]
    # Y_train = Y_train[1:200]
    
    ##***********************************************##
    ## Try Nearest Neighbors Classification ##
    ##***********************************************##
    from sklearn.neighbors import KNeighborsClassifier
    
    neigh = KNeighborsClassifier(n_neighbors = 10, weights = 'uniform', algorithm = 'auto',
                         leaf_size = 30, p = 2, metric = 'minkowski',
                         metric_params = None, n_jobs = 1)

    neigh.fit(X_train, Y_train)

    Y_train_predicted = neigh.predict(X_train)

    print(Y_train_predicted[0:20])
    # [ 1.  7.  4.  7.  9.  6.  4.  0.  7.  1.  8.  6.  8.  9.  2.  8.  6.  6.
    #   1.  6.]

    print()

    ## Get precision and recall on training data
    precision = dict()
    recall = dict()
    avg_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_train[:, i], Y_train_predicted[:,i])
        avg_precision[i] = average_precision_score(Y_train[:,i], Y_train_predicted[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_train.ravel(), Y_train_predicted.ravel())

    avg_precision["micro"] = average_precision_score(Y_train, Y_train_predicted, average = "micro")
    
    print("Avg precision score on train data", avg_precision["micro"]) # 1.0, 0.950670



    ## Get precision and recall on testing data
    Y_test_predicted = neigh.predict(X_test)
        
    precision = dict()
    recall = dict()
    avg_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], Y_test_predicted[:,i])
        avg_precision[i] = average_precision_score(Y_test[:,i], Y_test_predicted[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), Y_test_predicted.ravel())

    avg_precision["micro"] = average_precision_score(Y_test, Y_test_predicted, average = "micro")
    
    print("Avg precision score test", avg_precision["micro"]) # 0.9520681, 0.9471126

    print("Time elapsed", time.time() - start_time) # 4949.3, 6971.3

    
    # With n = 1, average test precision: 0.95206
    # average train precision = 1.0
    # Time elapsed: 4949 sec
    
    # With n = 10, average test precision: 0.94711
    # average train precision = 0.952068
    # Time elapsed: 6971 sec
