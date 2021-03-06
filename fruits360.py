import tensorflow as tf
import keras
from keras.preprocessing import image
from keras import optimizers
from keras.models import Model
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt 

# input train and test path

train_path = '../input/fruits-360_dataset/fruits-360/Training/'
val_path = '../input/fruits-360_dataset/fruits-360/Test/'


# Image Processing

train_datagen =  image.ImageDataGenerator(rescale = 1./255,
                                          shear_range = 0.2,
                                          zoom_range=0.2,
                                          rotation_range = 20)

validation_datagen = image.ImageDataGenerator(rescale = 1./255)


# Generators

train_generator = train_datagen.flow_from_directory(directory = train_path,
                                                    target_size = (50,50),
                                                    batch_size = 32,
                                                    color_mode = 'rgb',
                                                    class_mode='categorical',
                                                    shuffle = True)

val_generator = train_datagen.flow_from_directory(directory = val_path,
                                                    target_size = (50,50),
                                                    batch_size = 32,
                                                    color_mode = 'rgb',
                                                    class_mode='categorical',
                                                    shuffle = True)

# sequebtial model

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(50,50,3)))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.30))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(95, activation='softmax'))

print(model.summary())


# Compiling and fiting
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
history = model.fit_generator(train_generator,
                              epochs=20,
                              validation_data = val_generator)



#training and validation accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# training & validation loss values

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

