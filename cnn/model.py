import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

class WetterCNN:
    def __init__(self,data_dir):
        self.batch_size = 16
        self.img_height = 180
        self.img_width = 180

        self.data_dir = data_dir

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        class_names = self.train_ds.class_names
        print(class_names)

        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def build_model(self):
        # Definieren des Modells
        model = keras.Sequential(
            [
                layers.Input(shape=(self.img_height, self.img_width, 3)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(4, activation="softmax"),
            ]
        )
        # Kompilieren des Modells
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.save("model.h5")
        return model

    def train(self, epochs=5, batch_size=16):
        model = self.build_model()
        # Trainieren des Modells
        model.summary()
        model.fit(self.train_ds,validation_data=self.val_ds, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        self.model = model

    def evaluate(self):
        # Bewertung des Modells auf dem Testdatensatz
        test_loss, test_acc = self.model.evaluate(self.val_ds)
        print("Testverlust: ", test_loss)
        print("Testgenauigkeit: ", test_acc)

if __name__ == "__main__":
    cnn = WetterCNN(r"C:\Users\marlo\PycharmProjects\WETTER")
    cnn.train()
    cnn.evaluate()


