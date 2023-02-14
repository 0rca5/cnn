import tensorflow as tf
from keras.layers import Normalization
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt



class WeatherCNN:
    def __init__(self, data_dir, num_classes):
        self.model = None
        self.batch_size = 32
        self.img_height = 128
        self.img_width = 128
        self.num_classes = num_classes

        # Aufteilen der Daten in Trainings- und Validierungsdaten
        train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="both",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        # Normalisieren der Daten
        self.normalization_layer = self.__normalize_data(train_ds)

        # Optimieren der Daten
        self.train_ds = self.__optimize_training_data(train_ds)
        self.val_ds = self.__optimize_validation_data(val_ds)

    def __normalize_data(self, data):
        # Definieren der Normalisierungsschicht.
        normalization_layer = Normalization()
        # Passt Normalisierungswerte an die Bilddaten an. Verwendung von map(lambda x, _: x) da nur Bilddaten und nicht die labels benötigt werden.
        normalization_layer.adapt(data.map(lambda x, _: x))

        return normalization_layer

    def __optimize_training_data(self, data):
        """
         Optimiert Trainingsdaten, um Trainingsprozess zu beschleunigen.

        :param data: zu optimierender Datensatz
        :return: optimierter Datensatz
        """
        # Laden des Trainingsdatensatzes in einen Cache, um Ladezeiten zu minimieren
        train_ds = data.cache()
        # zufällige Mischung des Datensatzes mit einer Puffergröße von 1000,
        # damit Modell nicht auf eine bestimmte Reihenfolge trainiert wird
        train_ds = train_ds.shuffle(1000)
        # asynchrones Laden des nächsten Batches, während Modell auf aktuellem Batch trainiert.
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds

    def __optimize_validation_data(self, data):
        """
        Optimiert Validationsdaten, um Lernprozess zu beschleunigen.
        Unterschied zu __optimize_training_data(): keine shuffle(), da Reihenfolge der Daten für die Validierung
        nicht von Bedeutung ist.

        :param data: zu optimierender Datensatz
        :return: optimierter Datensatz
        """
        # Laden des Trainingsdatensatzes in einen Cache, um Ladezeiten zu minimieren
        val_ds = data.cache()
        # asynchrones Laden des nächsten Batches, während Modell auf aktuellem Batch trainiert.
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return val_ds


    def build_model(self):
        """Definiert eine CNN-Architektur"""

        self.model = keras.Sequential([
            layers.Input(shape=(self.img_height,self.img_width, 3)),
            self.normalization_layer,
            # padding = same stellt sicher, dass die Größe des Bildes nach der Faltung unverändert bleibt,
            # indem zusätzliche Pixel zum Ausgangsbild hinzugefügt werden
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Flatten-Methode wandelt mehrdimensionale Eingabe in 1D(Vektor)-Ausgabe um.
            # Größe des Vektors = Anzahl der Pixel*Farbkanäle.
            # Für einen einzelnen Wert im Vektor gilt: wert = höhe * breite * Anzahl Farbkanäle
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            # Verwendung einer Dropoutrate um Overfitting zu verhindern.
            layers.Dropout(0.3),
            layers.Dense(self.num_classes)
        ])
        return self.model

    def train(self, epochs=10):

        self.model = self.build_model()

        # Kompilieren des Modells
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        # Trainieren des Modells
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs
        )

        self.model.save("model.h5")

        return history

    def plot_training_process(self, history):
        # Plotten des Trainingsfortschritts
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(10)

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def evaluate(self):
        # Evaluieren des Modells auf dem Validierungsdatensatz
        test_loss, test_acc = self.model.evaluate(self.val_ds)
        print('Genauigkeit auf dem Validierungsdatensatz:', test_acc)


if __name__ == "__main__":
    cnn = WeatherCNN(r"C:\Users\marlo\PycharmProjects\weather_dataset_from_kaggle", 11)
    cnn.plot_training_process(cnn.train())
    cnn.evaluate()


