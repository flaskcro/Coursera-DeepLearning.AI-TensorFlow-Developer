import os
import shutil

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

class TransferLearn():
    def __init__(self, width, height, train_dir, test_dir):
        self.pretrained_model = None
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.width = width
        self.height = height
        self.train_dir = train_dir
        self.test_dir = test_dir

    def load_pretrained_model(self, local_weight_file):
        pretrained_model = InceptionV3(
            input_shape=(self.width, self.height, 3),
            include_top=False,
            weights=None
        )

        pretrained_model.load_weights(local_weight_file)
        for layer in pretrained_model.layers:
            layer.trainable = False
        self.pretrained_model = pretrained_model

    def struct_model(self, layer):
        last_layer = self.pretrained_model.get_layer(layer)
        last_output = last_layer.output
        x = layers.Flatten()(last_output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        model = Model(self.pretrained_model.input, x)
        model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def get_image_generator(self):
        train_datagen = ImageDataGenerator(rescale=1./255.,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        # Note that the validation data should not be augmented!
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

        # Flow training images in batches of 20 using train_datagen generator
        train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                            batch_size=20,
                                                            class_mode='binary',
                                                            target_size=(self.width, self.height))

        # Flow validation images in batches of 20 using test_datagen generator
        validation_generator = test_datagen.flow_from_directory(self.test_dir,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size=(self.width, self.height))
        self.train_generator = train_generator
        self.validation_generator = validation_generator

    def train(self):
        shutil.rmtree('./log')
        os.mkdir('./log')
        tensorboard_callback = TensorBoard(log_dir='./log', histogram_freq=1)

        history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            steps_per_epoch=100,
            epochs=20,
            validation_steps=50,
            callbacks=[tensorboard_callback],
            verbose=2)
        return history

if __name__ == '__main__':
    base_dir = '/tmp/cats_and_dogs_filtered'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    transferLearn = TransferLearn(150,150,train_dir,validation_dir)
    transferLearn.load_pretrained_model('/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    transferLearn.struct_model('mixed7')
    transferLearn.get_image_generator()
    history = transferLearn.train()