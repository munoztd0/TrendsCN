'''
Accompanying script 2 of the course 
"Trends in Computational Neuroscience"
michael.schartner@unige.ch 
'''

# Neural net classifier task: classifying 500 MNIST images
# Improve the classification performance of the current network (test accuracy: 0.75)  
# by changing something in this script or using a different classifier altogether.
# You may want to try changing the number of training epochs, width (nodes per layer), 
# depth (number of layers) and activation function (relu, elu, linear?).


import numpy as np
import os.path
from datetime import datetime

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2,l1
from keras.utils import np_utils
from keras import optimizers
#rm = optimizers.RMSprop(lr=0.001)
ada = optimizers.Adadelta() #change to an adapatative lr


def full_classifier_cross_validated():
    '''
    This function loads the first 500 MNIST
    data points, trains and tests a neural network
    classifier on 10 splits, and returns
    train and test accuracy
    ''' 

    # To time the whole function
    startTime=datetime.now()

    # Load data and check if data is already downloaded
    if os.path.exists('mnist_data.npy'):
        x = np.load('mnist_data.npy',allow_pickle=True)
        y1 = np.load('mnist_target.npy',allow_pickle=True)
    else:
        print('loading MNIST ...')
        mnist = fetch_openml("mnist_784", version=1)
        np.save('mnist_data.npy', mnist.data)
        np.save('mnist_target.npy', mnist.target)
        x=mnist.data
        y1=mnist.target.astype(int)
 
    y = np_utils.to_categorical(y1)
    
    # Set 
    #nodes1=10 #
    nodes2=35 #Bumbed from 35 to a 100
    EPOCHS=12 #added 12 epochs
    BATCH=20 #batch from 20 to 100

    # Preprocessing: Baseline-subtract, division by std
    #should I improve preprocess?
    sc = StandardScaler() 
    x = sc.fit_transform(x) 

    input_dim=len(x[0]) 
    k=0

    train_errs=[]
    test_errs=[]

    N_splits = 10
    kf = KFold(n_splits=N_splits,shuffle=True) 

    Q=[]
 
    # For each split of the data into train and test
    # a neural network model is created, trained and
    # used to classify unseen (i.e. test) datapoints
    for train_index, test_index in kf.split(x):

        train_X=x[train_index]
        test_X=x[test_index]
        train_y=y[train_index]
        test_y=y[test_index]

        model = Sequential() 
        
        #model.add(Dense(nodes1,activation='linear',input_shape=(input_dim,), kernel_regularizer=l2(0.01)))
        #model.add(Dropout(0.25)) #avoid to much overfitting
        #model.add(Dense(nodes2,activation='relu', kernel_regularizer=l2(0.01))) # add a reLu
        
        model.add(Dense(nodes2,activation='relu',input_shape=(input_dim,), kernel_regularizer=l2(0.01)))
        #model.add(Dense(nodes2,activation='relu', kernel_regularizer=l2(0.01))) # add a reLu
        #model.add(Dropout(0.5)) #avoid to much overfitting
        model.add(Dense(len(y[0]), input_dim=input_dim,activation='softmax',kernel_regularizer=l2(0.01)))
        #final node

        #model.compile(optimizer=rm,loss='categorical_crossentropy') 
        model.compile(optimizer=ada,loss='categorical_crossentropy') 
        if k==0: print(model.summary())
        print('training and testing, split %s of %s' %(k,N_splits))

        # The weights are adjusted, using the training set of datapoints
        model.fit(train_X, train_y, batch_size=BATCH, epochs=EPOCHS, verbose=0)

        # The unseen datapoints are classified by the trained network
        q=model.predict_on_batch(test_X)
        
        # The classification performance is measured in percentage correct
        o=[perf(y_true, y_pred) for y_true,y_pred in zip(test_y,q)]
        test_errs.append(o)

        # The classification performance is also recorded for the training data
        # to get a training error that may tell us that we have to train longer
        # if it is too high. Recall, accuracy = % correct, error = % incorrect 
        qq=model.predict_on_batch(train_X)
        oo=[perf(y_true, y_pred) for y_true,y_pred in zip(train_y,qq)]
        train_errs.append(oo)

        Q.append(o)   

        k+=1 

    train_errs2 = [item for sublist in train_errs for item in sublist]
    test_errs2 = [item for sublist in test_errs for item in sublist]

    print('Mean train accuracy: %s' %round(np.mean(train_errs2),4))
    STD = np.std(np.mean(test_errs,axis=1)) # standard deviation of accuracy across splits
    print('Mean test accuracy: %s +\- %s' %(round(np.mean(test_errs2),4), round(STD,2)))

    print("Epochs: ",EPOCHS)
    print("Duration: ",datetime.now()-startTime)

    K.clear_session() 


def perf(y_true,y_predict):
    '''
    This is a helper function, to check if
    y_true==y_predict; input is single sample, one-hot encoded
    '''
    if len(y_true)!=len(y_predict):
        return "lengths don't match"

    if np.argmax(y_true) == np.argmax(y_predict):   
        return 1
    else:
        return 0

#base                    Mean test accuracy: 0.76 +\- 0.04
#full : epoch 10 and 2 nodes Mean test accuracy: 0.868 +\- 0.03
#full : epoch 12 / 1 layers with 128 nodes / adadelta optim / 128 batch / Mean test accuracy: 0.9369 +\- 0.0
#full : epoch 12 / 100 batch / 2 layers with 35 node (1 linea & 1 relu) + 2 dropout layers  
#mean test accuracy: 0.9049 +\- 0.0
# without droupout layer 0.9254
#full : epoch 12 / 20 batch / 2 layers with a 15 linear node + a 35 relu node no droupout
#Mean test accuracy: 0.923 +\- 0.0
#full : epoch 12 / 20 batch / 2 layers with a 10 linear node + a 35 relu node no droupout
#Mean test accuracy: 0.9191 +\- 0.0
#full : epoch 12 / 20 batch / 1 layers with 35 relu nodes no droupout
#Mean test accuracy: 0.9326 +\- 0.0
#full : epoch 12 / 20 batch / 1 layers with 35 relu nodes WITH dropout
#Mean test accuracy: 0.9158 +\- 0.0