#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


# In[11]:


(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()
x_train = x_train.astype('float32') - 128.0
x_test = x_test.astype('float32') - 128.0
x_train  /= 128.0
x_test /= 128.0
y_train = to_categorical(y_train_)
y_test = to_categorical(y_test_)


# In[16]:


model = Sequential()
model.add(Conv2D(filters=6, 
                kernel_size=(5, 5),
                activation='relu',
                input_shape=(32, 32, 3)))
model.add(MaxPool2D())
model.add(Dropout(0.5))
model.add(Conv2D(filters=16,
                kernel_size=(5, 5),
                activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
print(model.summary())


# In[17]:


history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))


# In[18]:


import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['acc']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[19]:
model.save("keras-cifar10.h5")