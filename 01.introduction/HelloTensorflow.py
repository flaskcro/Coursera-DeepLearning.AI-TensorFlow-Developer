import tensorflow as tf
import numpy as np
from tensorflow import keras

#define simple network
#1 layer, 1 neuron in the layer, input shape is (1)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#y = 2x - 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype='float')
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype='float')

model.fit(xs, ys, epochs=500)

print(model.predict([5,6,7,8,9,10]))