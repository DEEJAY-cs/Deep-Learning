# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:49:30 2019

@author: dvo
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

#   Supress warning and informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class ConvNN(object):
    
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        self.num_classes=10
        self.batch_size=128
        self.epochs=24
        self.img_rows, self.img_cols=28,28

    def buildModel(self,input_shape):
        #   The Model
        model = Sequential()
        # add a Convalutional Layer to Neural Network
        model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
        # MaxPooling layer to 
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        # Flattens the dataset to provide a better matrix
        model.add(Flatten())
        # Fully connected layer with relu activation
        model.add(Dense(128,activation='relu'))
        # Dropout reduces the dataset in the network
        model.add(Dropout(0.5))
        # Fully connected layer with softmax activation
        model.add(Dense(self.num_classes,activation="softmax"))
        
        #   Compile
        #   Loss is determinded with categorical crossentropy
        #   Optimized with Adadelta
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
        return model

    def trainModel(self):
        (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
        input_shape=None
        if K.image_data_format()=='channel_first':
            x_train=x_train.reshape(x_train.shape[0],1,self.img_rows,self.img_cols)
            x_test=x_test.reshape(x_test.shape[0],1,self.img_rows,self.img_cols)
            input_shape=(1,self.img_rows,self.img_cols)
        else:
            x_train=x_train.reshape(x_train.shape[0],self.img_rows,self.img_cols,1)
            x_test=x_test.reshape(x_test.shape[0],self.img_rows,self.img_cols,1)
            input_shape=(self.img_rows,self.img_cols,1)
            x_train=x_train.astype('float32')/225
            x_test=x_test.astype('float32')/225
            y_train = keras.utils.to_categorical(y_train,self.num_classes)
            y_test = keras.utils.to_categorical(y_test,self.num_classes)
        model= self.buildModel(input_shape)
        # training 
        hist=model.fit(x_train,y_train,batch_size=self.batch_size,epochs=self.epochs,
                       verbose=1,validation_data=(x_test,y_test))
        # evaluate the train model on unseen dataset the test set
        score = model.evaluate(x_test,y_test,verbose=0)
        print("Test Loss: ",score[0])
        print("Test Accuracy: ",score[1])
        epoch_list= list(range(1,len(hist.history['acc'])+1))
        # plot epoch list to accuracy of training data and validation accuracy
        plt.plot(epoch_list,hist.history['acc'],epoch_list,hist.history['val_acc'])
        plt.legend(('Training Accuracy','Validation Accuracy'))
        plt.show()
        
    def run(self):
        self.trainModel()
            
if __name__ == "__main__":
    convNN=ConvNN()
    convNN.run()