"""
Description:
    Train any neural networks and compare performance
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def view_data(x):
    print(x.shape)
    plt.imshow(x, cmap = plt.cm.binary)
    plt.show()
    return

def train_tf_model(x_train, y_train):
    """ Train and return MLP """
    # Create Basic Sequential Model - MLP
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    
    # Train model
    model.fit(x_train, y_train, epochs=3)

    return model

def main():
    # Load MNIST dataset
    dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Normalise Inputs for Training and Test Data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # View some MNIST images
    for i in range(5):
        print(f"Target value: {y_train[i]}")
        view_data(x_train[i])

    # Train a TensorFlow Model for Baseline Performance
    tf_model = train_tf_model(x_train, y_train)

    # Test models and print results
    test_loss, test_accuracy = tf_model.evaluate(x_test, y_test)
    print(test_loss, test_accuracy)

    # Use model to get predictions
    predictions = tf_model.predict(x_test)
    for i in range(10):
        print(np.argmax(predictions[i]), y_test[i])
    

if __name__ == "__main__":
    main()