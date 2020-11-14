#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv("C:/Users/preer/Desktop/digit recogniser/digit-recognizer/train.csv")
test  = pd.read_csv("C:/Users/preer/Desktop/digit recogniser/digit-recognizer/test.csv")

y_train = train["label"]

x_train = train.drop(["label"],axis=1)

del train 


##vis


# In[5]:


##visualize count plot & value of all the counts 

g = sns.countplot(y_train)

y_train.value_counts()


# In[7]:


## null value detection :

x_train.isnull().sum()


# In[8]:


## normalizing the pixel:

x_train = x_train/255.0
test    = test/255.0

## reshape the values into 3D values as keras requires an extra dimention:

x_train = x_train.reshape(-1,28,28,1)
test    = test.reshape(-1,28,28,1)

y_train = to_categorical(y_train , num_classes = 10)


# In[ ]:



x_train.shape

##split training & validation sets 

random_seed = 2

##split train & test set 

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=2)

print(x_train.shape,y_train.shape,x_val,y_val)


# In[42]:


## plot show :

g = plt.imshow(x_train[0][:,:,0])


# In[ ]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[28]:


optimizer = RMSprop(lr=.001,rho=.9,epsilon=le-08,decay=0.0)
model.compile(opitimizer = optimizer,loss= "categorical_crossentrophy",metrics=["accuracy"])


# In[29]:


## Reduce the LR to half of what it was if accuracy is not improved by 3 epochs (factor=1/2,patience or epochs= 3)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=.5,min_lr=.00001)


# In[ ]:


epochs = 1 
batch_size = 86

# Without data augmentation i obtained an accuracy of 0.98114
#history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 
#          validation_data = (x_val, y_val), verbose = 2)


# In[24]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


# In[30]:


datagen.fit(x_train)


# In[16]:


history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:




