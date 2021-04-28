import tensorflow as tf
from os import path, getcwd

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') > 0.998:
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    my_callback = MyCallback()
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    data_path = f"{getcwd()}/data/mnist.npz"
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=data_path)
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model2 = tf.keras.models.Sequential([
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        training_images, training_labels, epochs=20, callbacks=[my_callback]
        # YOUR CODE ENDS HERE
    )

    print(model.evaluate(test_images, test_labels))

    model.save('./model/fashion_mnist_cnn')
    return history.epoch, history.history['accuracy'][-1]


_, _ = train_mnist_conv()
