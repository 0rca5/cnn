import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

class WetterCNN:
    def __init__(self):
        

    def build_model(self):
        # Definieren des Modells
        model = keras.Sequential(
            [
                layers.Input(shape=(28, 28, 1, 1)),
                layers.Conv3D(32, kernel_size=(3, 3, 3), activation="relu"),
                layers.MaxPooling3D(pool_size=(2, 2, 2)),
                layers.Conv3D(64, kernel_size=(3, 3, 3), activation="relu"),
                layers.MaxPooling3D(pool_size=(2, 2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="softmax"),
            ]
        )
        # Kompilieren des Modells
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train(self, epochs=5, batch_size=128):
        model = self.build_model()
        # Trainieren des Modells
        model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        self.model = model

    def evaluate(self):
        # Bewertung des Modells auf dem Testdatensatz
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("Testverlust: ", test_loss)
        print("Testgenauigkeit: ", test_acc)


if __name__ == "__main__":
    cnn = WetterCNN()
    cnn.train()
    cnn.evaluate()

