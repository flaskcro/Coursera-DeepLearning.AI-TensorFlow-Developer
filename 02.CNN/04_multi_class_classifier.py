import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os


def get_data(filename):
    with open(filename) as training_file:
        content = np.loadtxt(filename, skiprows=1, delimiter=",")
        labels = content[:, 0]
        images = content[:, 1:].reshape(-1, 28, 28)
    return images, labels

path_sign_mnist_train = "/tmp/sign_mnist_train.csv"
path_sign_mnist_test = "/tmp/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow(
    training_images,
    training_labels,
    batch_size=128
)

validation_generator = validation_datagen.flow(
    testing_images,
    testing_labels,
    batch_size=128
)

model = tf.keras.models.Sequential([
    # Your Code Here
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(25, activation='softmax')
])

# Compile Model.
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=100,
                    callbacks=[EarlyStopping(patience=5, monitor='val_loss'), TensorBoard(log_dir='./log', histogram_freq=1)],
                    verbose=2)

if not os.path.isdir('model'):
    os.mkdir('model')

if not os.path.isdir('model/sign_mnist'):
    os.mkdir('model/sign_mnist')

print(model.evaluate(testing_images, testing_labels, verbose=0))
model.save('model/sign_mnist')