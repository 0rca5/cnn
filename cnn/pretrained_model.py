import tensorflow as tf
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import ResNet152V2


class KerasCNN:
    def __init__(self, data_dir, num_classes):
        self._model = None
        self.batch_size = 32
        self.img_height = 224
        self.img_width = 224
        self.num_classes = num_classes

        # Data Augmentation
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           rotation_range=30,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Aufteilen der Daten in Trainings- und Validierungsdaten
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training')

        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation')

        # Laden des vortrainierten Modells
        resnet_model = ResNet152V2(weights='imagenet',
                                   include_top=False,
                                   input_shape=(self.img_height, self.img_width, 3))
        # Freezing des Modells
        for layer in resnet_model.layers:
            layer.trainable = False

        # Hinzufügen von Schichten für Transfer Learning
        x = resnet_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)

        self._model = models.Model(inputs=resnet_model.input, outputs=predictions)

        # Kompilierung
        self._model.compile(optimizer="Adam",
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

        # Training des Modells
        self._model.fit(train_generator,
                        epochs=10,
                        validation_data=val_generator)

    def load_model(self, model):
        self._model = model

    def get_model(self):
        if self._model:
            return self._model
        else:
            raise Exception("class KerasCNN has no model. Try the load_model method first.")


if __name__ == "__main__":
    cnn = KerasCNN(r"C:\Users\marlo\PycharmProjects\weather_dataset_from_kaggle", 11)
    print(cnn.get_model().summary())
    tf.keras.models.save_model(cnn.get_model(), "fine_tuned_model.h5")
