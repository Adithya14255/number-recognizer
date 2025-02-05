import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Define a simple Neural Network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28,1)),  # Flatten 28x28 images
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # Output layer (10 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train the model
model.fit(x_train, y_train, epochs=100, batch_size=64)

# 4. Save the trained model
model.save("model.h5")
# OR Save in TensorFlow SavedModel format
# model.save("saved_model")

print("Model saved as model.h5")