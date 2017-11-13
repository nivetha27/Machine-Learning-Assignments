import numpy as np
import os

filePath = 'C:/Users/nsathya/Desktop/CSE546P-ML/Machine-Learning-Assignments/Assignment 3/MNIST_PCA/'
imageFileName = filePath + 'train-images-pca.idx2-double'
labelFileName = filePath + 'train-labels.idx1-ubyte'
f = open(imageFileName, "rb")
f.seek(12, os.SEEK_SET)
arr = np.fromfile(f, dtype=np.float);
print arr[0];