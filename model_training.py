import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# Load the data
data = pd.read_pickle(os.path.abspath(os.getcwd())+'/content/dataset.pkl')
X = np.array([np.asarray(x) for x in data['Graph'].values])
Y = np.array([np.asarray(x) for x in data['Degree'].values])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Add a channel dimension
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6)  # 6 output classes (degrees from 0 to 5)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), callbacks=[early_stopping])

# Save the model
model.save(os.path.abspath(os.getcwd())+'/content/model.h5')

# Plot the training and validation loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.abspath(os.getcwd())+'/content/loss.png')
plt.close()

# Plot the training and validation accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.abspath(os.getcwd())+'/content/accuracy.png')
plt.close()

# Plot some predictions
predictions = model.predict(X_test)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i, :, :, 0], cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {Y_test[i]}")
    plt.axis('off')
plt.savefig(os.path.abspath(os.getcwd())+'/content/predictions.png')
plt.close()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss:.2f}")
print(f"Test Accuracy: {accuracy:.2f}")
