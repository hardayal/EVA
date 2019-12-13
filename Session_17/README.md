# TensorFlow Profiler
Tensorflow has a built-in profiler called TensorBoard that allows you to visualise & record each operations & metrics values while training the model. This profile results can be visualised using TensorBoard's profile plug-in.

# 1.Prequisites
* First step is to install TensorBoard in your local machine.
* This tutorial is written in consideration of notebook in Google colab. In colab notebook select "GPU" in "Notebook setting".

![Notebook setting](https://raw.githubusercontent.com/tensorflow/tensorboard/master/docs/images/profiler-notebook-settings.png)
# 2. Set up

To enable TensorBoard in colab, write the below code in colab notebook.

```
%load_ext tensorboard
```

# 3. Display GPU

Below code will display the GPU for TensorFlow.

```
device_name = tf.test.gpu_device_name()
if not tf.test.is_gpu_available():
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

# 4. Create a simple model

In this step you are going to run a simple keras model using TensorBoard callback. Below is an example code in keras.

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
​
from datetime import datetime
from packaging import version
​
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
​
import numpy as np
​
print("TensorFlow version: ", tf.__version__)
​
device_name = tf.test.gpu_device_name()
if not tf.test.is_gpu_available():
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
​
# Importing all necessary keras Modules
from keras import backend as K
import time
import matplotlib.pyplot as plt
import numpy as np
% matplotlib inline
np.random.seed(2017) 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
​
# Download the CIFAR10 dataset
from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  test_features.shape
num_classes = len(np.unique(train_labels))
​
# Visualization of images from dataset
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(train_labels[:]==i)[0]
    features_idx = train_features[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = features_idx[img_num]
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()
​
# Function defined for plotting the model history i.e epoch vs validation accuracy & epoch vs validation loss
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
​
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)
​
# As shown above the label matrix(train_features & test_features) holds the images. So, keras may find some relational order
# between these values. To avoid this problem we hot encode the matrix into a binary matrix. This matrix has number of columns equal to the number
# of classes(10 columns in this scenario). Each row defines the label of one sample point in data set & has only one '1' & others are '0'.
# Convert 1-dimensional class arrays to 10-dimensional class matrices
train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255
# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)
​
# Defined function for learning rate schedule
def scheduler(epoch,lr):
  decay = 0.1
  return round(0.01 * 1/(1 + decay * epoch) , 10)
​
# Define the model
model = Sequential()
​
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
​
model.add(Convolution2D(32, 1, 1, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
​
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3 , border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
​
model.add(Convolution2D(10, 1, 1, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
​
model.add(Convolution2D(10, 8, 8))
​
model.add(Flatten())
​
model.add(Activation('softmax'))
​
# Compile the model
model.compile(optimizer=Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
​
# summary of the model
model.summary()
```

# 5.Create TensorBoard Callback

Below code is used for creating a Tensorboard callback.

```
log_dir="logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")
​
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3)
# checkpoint. Saves the best model
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint , LearningRateScheduler(scheduler,verbose = 1) , tensorboard_callback]
```

* log_dir is the path of the directory where the log files will be saved
* histogram_freq is the frequency in epochs at which the activations & weight histograms are computed for the layers of the model. If 0 histograms will not be computed. validation data or split must be there for histogram visualisation.
* profile_batch is used for profile the batch to sample compute characteristics.

# 6.Train the model

Train the keras model.

```
# train the model
start = time.time()
# Train the model
model_info = model.fit(train_features, train_labels, batch_size = 128,
                                 nb_epoch = 20, 
                                 validation_data = (test_features,test_labels), callbacks = callbacks_list)
end = time.time()
print ("Model took %0.2f seconds to train"%(end - start))
# plot model history
plot_model_history(model_info)
# compute test accuracy
print ("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model))
```

# 7.Visualise profile using TensorBoard

TensorBoard profiles can not be visualised in colab. This log file needs to be downloaded in local machine & then to be visualised in TensorBoard. The log file first needs to be zipped & then download in local machine. Use below code for compressing the log file.

```
!tar -zcvf logs.tar.gz logs/profile/
cd download/directory
> tar -zxvf logs.tar.gz
> tensorboard --logdir=logs/ --port=6006
```
Download logdir.tar.gz by right-clicking it in “Files” tab.

![Download File](https://raw.githubusercontent.com/tensorflow/tensorboard/master/docs/images/profiler-download-logdir.png)

# 8.Visualise profile

Open a new tab in your Chrome browser and navigate to [localhost:6006](http://localhost:6006/) and then click "Profile" tab.

![Profile Screenshots 1](https://github.com/hardayal/EVA/blob/master/Session_17/Visualise_Profile_1.png)

![Profile Screenshots 2](https://github.com/hardayal/EVA/blob/master/Session_17/Visualise_Profile_2.png)
