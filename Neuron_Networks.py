import tensorflow as tf
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
print("Enter image directory")
image_directory_conv = input()
image_directory_conv = image_directory_conv.reshape((10000, 150, 150, 1))
image_directory_conv= image_directory_conv.astype('float32') / 255
print("Enter image directory test")
image_directory_test = input()
image_directory_conv = image_directory_conv.reshape((1000, 150, 150, 1))
image_directory_conv= image_directory_conv.astype('float32') / 255
fig, ax = plt.subplots(1, 5, figsize =(15, 10))
def load_and_preprocess_image_data(image_directory_conv):       
    # Создание генератора для загрузки изображений
    num_images_to_load = 100
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    image_data = image_generator.flow_from_directory(
    image_directory_conv,
    target_size=(150, 150),
    batch_size=num_images_to_load,
    class_mode='binary')
    return image_data, image_data.class_mode       
# Load the image data
# Replace this with code to load and preprocess your image data
X_train, y_train = load_and_preprocess_image_data(image_directory_conv)

# Define the TensorFlow model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
def load_and_preprocess_test_image(image_directory_test):
    num_images_to_load = 1  # Load only one test image
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_image_data = image_generator.flow_from_directory(
        image_directory_test,
        target_size=(150, 150),
        batch_size=num_images_to_load,
        class_mode='binary',
        shuffle=False)  # Set shuffle to False to ensure the order of test images
    return test_image_data[0][0][0]  # Return the first test image
# Make predictions
# Replace this with code to load and preprocess your test image
test_image = load_and_preprocess_test_image(image_directory_test)
predicted_label = model.predict(np.array([test_image]))

# Print the predicted label
if predicted_label > 0.5:
    print("Predicted image label: Class A")
else:
    print("Predicted image label: Class B")