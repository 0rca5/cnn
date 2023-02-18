import tensorflow as tf
from keras.applications import ResNet152V2
from keras.layers import Normalization
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from keras_tuner.tuners import RandomSearch
import numpy as np


class WeatherCNN:
    def __init__(self, data_dir, num_classes):
        self._model = None
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
        """
        Erstellt eine Normalisierungsschicht, dessen Parameter an die übergebenen Daten angepasst sind.
        :param data: Daten, an den der Normalisation-Schicht angepasst werden soll.
        :return: Normalisation-Schicht
        """
        # Definieren der Normalisierungsschicht.
        normalization_layer = Normalization()
        # Passt Normalisierungswerte(Mittelwert & Varianz) an die Bilddaten an.
        # Verwendung von map(lambda x, _: x) da nur Bilddaten und nicht die labels benötigt werden.
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

    def build_model_for_hyperparameter_search(self, hp):
        """Definiert eine CNN-Architektur für die Suche nach den optimalen Hyperparametern"""

        self._model = keras.Sequential([
            layers.Input(shape=(self.img_height,self.img_width, 3)),
            self.normalization_layer,
            # padding = same stellt sicher, dass die Größe des Bildes nach der Faltung unverändert bleibt,
            # indem zusätzliche Pixel zum Ausgangsbild hinzugefügt werden
            layers.Conv2D(hp.Int('conv1_units', min_value=8, max_value=64, step=8), 3, padding='same',
                          activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(hp.Int('conv2_units', min_value=16, max_value=128, step=16), 3, padding='same',
                          activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(hp.Int('conv3_units', min_value=32, max_value=256, step=32), 3, padding='same',
                          activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            # Flatten-Methode wandelt mehrdimensionale Eingabe in 1D(Vektor)-Ausgabe um.
            # Größe des Vektors = Anzahl der Pixel*Farbkanäle.
            # Für einen einzelnen Wert im Vektor gilt: wert = höhe * breite * Anzahl Farbkanäle
            layers.Flatten(),
            layers.Dense(hp.Int('dense_units', min_value=32, max_value=512, step=32), activation='relu'),
            # Verwendung einer Dropoutrate um Overfitting zu verhindern.
            layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.2)),
            layers.Dense(self.num_classes)
        ])

        self._model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
        return self._model

    def build_standard_model(self):
        """Definiert eine CNN-Architektur"""

        self._model = keras.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 3)),
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
            layers.Dense(self.num_classes, activation="softmax")
        ])

        print(self._model.summary())
        return self._model

    def build_pretrained_model(self, pretrained_model):

        # Freezing des Modells
        for layer in pretrained_model.layers:
            layer.trainable = False

        # Hinzufügen von Schichten für Transfer Learning
        x = pretrained_model.output
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        predictions = layers.Dense(self.num_classes, activation="softmax")(x)

        self._model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)

        return self._model

    def search_optimal_hyperparameter(self, epochs=3, executions_per_trial=1, max_trials=20):
        """
        führt eine Suche nach den optimalen Hyperparametern durch

        :param epochs: Anzahl der Epochen die eine Hyperparameter-Kombination (CNN-Architektur) trainiert werden soll
        :param executions_per_trial: Anzahl an Trainingsversuchen pro Hyperparameter-Kombination (CNN-Architektur)
        :param max_trials: Maximale Anzahl an Hyperparameter-Kombinationen (CNN-Architekturen)
        """

        # Hyperparameter-Suche
        tuner = RandomSearch(
            self.build_model_for_hyperparameter_search,
            # Zielwert, anhand dessen Modelle verglichen werden
            objective='val_accuracy',
            # Maximale Anzahl an Hyperparameter-Kombinationen, also verschiedenen Modellen
            max_trials=max_trials,
            # Anzahl der Durchgänge pro Hyperparameter-Kombination (erreichter Maximalwert wird genommen)
            executions_per_trial=executions_per_trial,
            # Ort an dem die Ergebnisse gespeichert werden
            directory='tuner',
            project_name='weather_classification',
            overwrite=True,
        )

        tuner.search_space_summary()

        tuner.search(self.train_ds, epochs=epochs, validation_data=self.val_ds)

        # Ausgabe der Ergebnisse
        print("Ergebnisse der Suche:")
        tuner.results_summary()

        # Beste Konfiguration verwenden
        self._model = tuner.get_best_models(num_models=1)[0]

        return self._model

    def train(self, epochs=10, optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy()):
        """
        trainiert ein CNN.
        :param epochs: Anzahl der Trainingsdurchgänge
        :param optimizer: Optimierer
        :param loss: Verlustfunktion
        :return: Trainingsprozess (model.fit())
        """
        if not self._model:
            self._model = self.build_standard_model()

        # Kompilieren des Modells
        self._model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # Trainieren des Modells
        history = self._model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            batch_size=32
        )

        self._model.save("best_model.h5")

        return history

    def plot_training_process(self, history):
        """
        stellt den Trainingsprozess grafisch dar.
        :param history: Trainingsprozess (model.fit())
        :return: None
        """
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
        test_loss, test_acc = self._model.evaluate(self.val_ds)
        print('Genauigkeit auf dem Validierungsdatensatz:', test_acc)

    def predict(self, data, verbose=2):
        """
        Liefert eine Vorhersage anhand des Models für ein oder mehrere Bild(er)
        :param data: (String oder tf.dataset) Bild(er), für die es eine Vorhersage zu treffen gilt
        :param verbose: Sichtbarkeit
        :return: (int oder list(int)) vorhergesagtes Label
        """

        # Für ein Bild. Muss als URL-String übergeben wird
        if isinstance(data, str):
            img = keras.preprocessing.image.load_img(data, target_size=(self.img_height, self.img_width))
            img_array = keras.preprocessing.image.img_to_array(img)

            # Fügt eine neue Dimension an den Anfang von img_array.
            img_array = np.expand_dims(img_array, 0)
            # img_array hat jetzt die Form (1,self.img_height,self.img_width)

            normalized_img = self.normalization_layer(img_array)

            prediction = self._model.predict(normalized_img, verbose=verbose)

            # Liefert die Klasse mit der höchsten Wahrscheinlichkeit
            predicted_class = tf.argmax(prediction[0]).numpy()

            return predicted_class

        # Für mehrere Bilder. Muss in Form eines tensorflow-datasets vorliegen
        else:
            prediction = self._model.predict(data, verbose=verbose)
            predicted_classes = tf.argmax(prediction, axis=1).numpy()

            return predicted_classes

    def show_feature_map(self, layer_name, filter_index, url):
        """
        zeigt die Feature-Map für einen spezifischen filter in einem spezifischen convolutional layer

        :param layer_name: Name des convolutional layer
        :param filter_index: index des filters
        :param url: URL des Eingabe-Bilds
        :return: None
        """
        # Ausgabetensor der spezifischen Schicht
        layer_outputs = [layer.output for layer in self._model.layers if layer.name == layer_name]
        if not layer_outputs:
            print(f"No layer with id {layer_name} found in model")
            return

        # Erstellen eines (Teil-)Modells des ursprünglichen Modells, wobei der Output die spezifische Schicht ist.
        activation_model = tf.keras.models.Model(inputs=self._model.input, outputs=layer_outputs)

        # Vorverarbeitung des Eingabebilds
        img_path = url
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img_tensor = tf.keras.preprocessing.image.img_to_array(img)
        # Resizing
        img_tensor = tf.keras.preprocessing.image.smart_resize(img_tensor, (self.img_height, self.img_width))
        img_tensor = img_tensor.reshape((1, self.img_height, self.img_width, 3))
        # Normalisierung
        img_tensor = self.normalization_layer(img_tensor)

        # Erstellen der Feature-Maps für das Eingabebild
        activations = activation_model.predict(img_tensor)

        feature_map = activations[0][:, :, filter_index]
        plt.matshow(feature_map, cmap='gray')
        plt.show()

    def show_all_feature_maps(self, layer_name, url):
        """
        zeigt alle Feature-Maps eines spezifischen convolutional layers

        :param layer_name: Name des convolutional layer
        :param url: URL des Eingabebilds
        :return: None
        """

        # Ausgabetensor der spezifischen Schicht
        layer_outputs = [layer.output for layer in self._model.layers if layer.name == layer_name]
        if not layer_outputs:
            print(f"No layer with name {layer_name} found in model")
            return

        # Erstellen eines (Teil-)Modells des ursprünglichen Modells, wobei der Output die spezifische Schicht ist.
        activation_model = tf.keras.models.Model(inputs=self._model.input, outputs=layer_outputs)

        # Vorverarbeitung des Eingabebilds
        img_path = url
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img_tensor = tf.keras.preprocessing.image.img_to_array(img)
        # Resizing
        img_tensor = tf.keras.preprocessing.image.smart_resize(img_tensor, (self.img_height, self.img_width))
        img_tensor = img_tensor.reshape((1, self.img_height, self.img_width, 3))
        # Normalisierung
        img_tensor = self.normalization_layer(img_tensor)

        # Erstellen der Feature-Maps für das Eingabebild
        activations = activation_model.predict(img_tensor)

        # Konkatenieren der einzelnen Feature-Maps entlang der Tiefenachse: Tiefe entspricht der Anzahl an Filtern
        feature_maps = np.concatenate(activations, axis=-1)

        # Anzeigen der Feature-Maps, -1 für Tiefenkanal: Anzahl Feature-Maps
        num_feature_maps = feature_maps.shape[-1]
        rows = int(np.ceil(np.sqrt(num_feature_maps)))
        cols = int(np.ceil(num_feature_maps / rows))
        # axs = Array der Feature-Maps
        anzeige_fenster, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
        anzeige_fenster.suptitle(layer_name)

        for i in range(num_feature_maps):
            row = i // cols
            col = i % cols
            axs[row, col].matshow(feature_maps[:, :, i], cmap='gray')
            axs[row, col].axis('off')

        plt.show()

    def load_model(self, model_to_load):
        self._model = model_to_load

    def get_model(self):
        if self._model:
            return self._model
        else:
            raise Exception("class CNN has no model. Try the load_model method first.")


if __name__ == "__main__":
    cnn = WeatherCNN(r"C:\Users\marlo\PycharmProjects\weather_dataset_from_kaggle", 11)

    #cnn.build_pretrained_model(ResNet152V2(weights='imagenet', include_top=False, input_shape=(cnn.img_height, cnn.img_width, 3)))
    # cnn.search_optimal_hyperparameter()
    # cnn.plot_training_process(cnn.train())
    # cnn._model = tf.keras.models.load_model("best_model.h5")
    # cnn.show_all_feature_maps("conv2d", r"C:\Users\marlo\PycharmProjects\weather_dataset_from_kaggle\lightning\1853.jpg")

    model = tf.keras.models.load_model("best_model.h5")
    model.summary()

















