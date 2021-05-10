from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

class CatsAndDogTrainer:

    def __init__(self, train_dir, valid_dir):
        self.model = None
        self.train_dir = train_dir
        self.validation_dir = valid_dir

    def make_model(self):
        model = Sequential([
            Conv2D(16,(3,3), activation='relu', input_shape=(150,150,3)),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=RMSprop(lr=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def image_load(self):
        train_datagen = ImageDataGenerator(
            rescale=1./ 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            batch_size=10,
                                                            class_mode='binary',
                                                            target_size=(150, 150))


        validation_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        validation_generator = validation_datagen.flow_from_directory(self.validation_dir,
                                                                      batch_size=10,
                                                                      class_mode='binary',
                                                                      target_size=(150, 150))

        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def train(self):
        self.make_model()
        self.image_load()
        history = self.model.fit_generator(self.train_generator,
                                      epochs=10,
                                      verbose=1,
                                      validation_data=self.validation_generator)

        return history

if __name__ == '__main__':
    trainer = CatsAndDogTrainer("/tmp/cats-v-dogs/training", "/tmp/cats-v-dogs/testing")
    history = trainer.train()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    print(acc)
    print(val_acc)