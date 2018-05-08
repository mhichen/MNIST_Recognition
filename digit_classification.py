import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

    # Check a few images
    #show_image(X, Y, 56000, (28, 28))

    ## Split data into train, test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 8012, shuffle = True)

