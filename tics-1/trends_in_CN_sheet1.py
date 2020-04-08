'''
Accompanying script 1 of the course 
"Trends in Computational Neuroscience"

michael.schartner@unige.ch 
'''

# Task A: Dimesionality reduction of the MNIST_784 dataset
# with the UMAP algorithm.
# If you are new to python, 
# try to run this script line by line 
# in an interactive python session (ipython).
# If a python package is missing, say it's called "pack", try installing it via 'pip install pack' 
# in a terminal.

# Modified from https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_mnist_example.html#sphx-glr-auto-examples-plot-mnist-example-p

#conda create -n trendsCN python=3.7
#conda activate trendsCN
#conda install ipython
#pip install numpy scipy==1.4.1 matplotlib seaborn scikit-learn tensorflow keras 
#conda install -c conda-forge umap-learn

#import required libraries
import numpy as np
import pandas as pd
import umap
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib
import os
plt.ion() #switch on interactive plotting

# set some plotting option
sns.set(context="paper", style="white")

# download the MNIST_784 dataset of hand-writted digets
# (this takes a minute or so)
#mnist = fetch_openml("mnist_784", version=1)

# Let's save part of the data locally to 
# access it more quickly later, when tweaking your code
if os.path.exists('mnist_data10000.npy'):
    data = np.load('mnist_data10000.npy',allow_pickle=True)
    target = np.load('mnist_target10000.npy',allow_pickle=True)
else:
    print('loading MNIST ...')
    mnist = fetch_openml("mnist_784", version=1)
    np.save('mnist_data10000.npy', mnist.data[:10000])
    np.save('mnist_target10000.npy', mnist.target[:10000])
    data = np.load('mnist_data10000.npy',allow_pickle=True)
    target = np.load('mnist_target10000.npy',allow_pickle=True)

# Plot image number 1000 to exemplify a data point and check label
fig0, ax0 = plt.subplots()
ax0.set_title('Example data point - label: %s' %target[1000])
plt.imshow(data[1000].reshape([28,28]), axes = ax0)
plt.show()

# Apply the umap algorithm to reduce the 28**2-dimensional points to 2 D
# (this takes a minute or so; to speed it up we'll only use 10k datapoints, not all 70k)
reducer = umap.UMAP(random_state=42)
print('Embedding ...')
embedding = reducer.fit_transform(data)


# Getting the data-labels as a number for color
color = target.astype(int)

# Plot the results
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

# construct a legend manually
cmap = matplotlib.cm.get_cmap('Spectral')
norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
digits = np.unique(color)
Patches = [mpatches.Patch(color=cmap(norm(d)), label=d) for d in digits]
plt.legend(handles=Patches)
plt.show()

# Does it make sense? Are clusters that are close together in 2d 
# also hand-written digets that look alike?
# 3,5,8  & 4,9,7,

# UMAP tasks:
# 1a: Plot only odd digits.
colorODD = (color[color%2!=1])
dataODD = (data[color%2!=1])
embeddingODD = reducer.fit_transform(dataODD)


# Plot the results
fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(embeddingODD[:, 0], embeddingODD[:, 1], c=colorODD, cmap="rainbow", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data ODD numbers embedded \n into two dimensions by UMAP", fontsize=18)

# construct a legend manually
cmap = matplotlib.cm.get_cmap('rainbow')
norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
digits = np.unique(color)
Patches = [mpatches.Patch(color=cmap(norm(d)), label=d) for d in digits]
Patches = [ Patches[i] for i in [0, 2, 4, 6, 8] ]
plt.legend(handles=Patches)
plt.show()

# 1b: Create and describe a UMAP plot for any non-MNIST dataset you find in the net.

iris = load_iris()
embeddingIRIS = reducer.fit_transform(iris.data)
#embedding.shape


# Plot the results
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(embeddingIRIS[:, 0], embeddingIRIS[:, 1], c=iris.target, cmap="Set2", s=5)
plt.setp(ax, xticks=[], yticks=[])
plt.title("IRIS data embedded \n into two dimensions by UMAP", fontsize=13)

# construct a legend manually
cmap = matplotlib.cm.get_cmap('Set2')
norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
digits = np.unique(iris.target)
lab = np.unique(iris.target_names)
Patches = [mpatches.Patch(color=cmap(norm(d)), label=lab[d]) for d in digits]
plt.legend(handles=Patches)
plt.show()

