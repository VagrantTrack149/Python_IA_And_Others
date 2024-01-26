import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create the neural network
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
model.evaluate(x_test, y_test)

# correct this
# Predict the number of an image
#image = tf.keras.utils.load_img('D:/Descargas/MNIST_6_0.png', target_size=(28, 28))
#image = tf.keras.utils.img_to_array(image)
#image = image / 255.0
#image = image.reshape(28, 28)
#prediction = model.predict(image)
#print(prediction)
