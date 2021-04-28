import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

tf.random.set_seed(42)

dataset = pd.read_csv('./data/diabetes.csv')
X = dataset.iloc[:,0:8]
scaler = MinMaxScaler()
X.iloc[:,1:7] = scaler.fit_transform(X.iloc[:,1:7])

X = X.to_numpy()
y = dataset[['Outcome']].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = keras.Sequential([
    keras.layers.Dense(12, input_dim=8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    #keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, callbacks=[early_stopping_callback])

print(model.evaluate(X_test, y_test))
model.save('./model/pima_indian_diabetes')
