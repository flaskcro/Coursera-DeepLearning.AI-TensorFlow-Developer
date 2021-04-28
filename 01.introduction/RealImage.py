from tensorflow import keras


class MyCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        desired_accuracy = 0.999
        if logs.get('accuracy') > desired_accuracy:
            print("\nReached 99.9% accuracy")
            self.model.stop_training = True


class RealImageTrain:

    def __init__(self, directory):
        self.directory = directory
        self.model = None
        self.train_generator = None

    def construct_model(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            #keras.layers.Conv2D(64, (3, 3), activation='relu'),
            #keras.layers.MaxPooling2D(2, 2),
            #keras.layers.Conv2D(64, (3, 3), activation='relu'),
            #keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy',
                      optimizer=keras.optimizers.RMSprop(lr=0.001),
                      metrics=['accuracy']
                      )

        self.model = model

    def data_gen(self):
        print(self.directory)
        train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
        train_generator = train_data_gen.flow_from_directory(
            self.directory,
            target_size=(150, 150),
            batch_size=16,
            class_mode='binary'
        )
        self.train_generator = train_generator

    def build_model(self):
        self.construct_model()
        self.data_gen()
        my_callback = MyCallBack()

        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=5,
            epochs=30, verbose=1, callbacks=[my_callback])
        self.model.save('./model/happy_or_sad')
        print(history.history['accuracy'][-1])


if __name__ == '__main__':
    real_image_Train = RealImageTrain('data/happy_or_sad/')
    real_image_Train.build_model()
    #print(real_image_Train.model.summary())
