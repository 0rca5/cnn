import shutil
import os
import urllib.request
import requests
import json
from Wetterdienst_Service import WetterdienstService


class PreprocessingService:

    _instance = None
    _service = None

    def __init__(self):
        raise RuntimeError("Nutze stattdessen instance()-Methode")

    @classmethod
    def instance(cls):
        """
        Sorgt dafür, dass immer die gleiche WetterdienstService-Instanz referenziert wird
        und initialisert eine WetterdienstPersistence-Instanz für alle Methoden.

        :return: Eine WetterdienstService-Instanz
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._service = WetterdienstService.instance()
        return cls._instance


    def filter_meldungen_after_category_and_image_url(cls, url):
        """
        filtert die crowd_meldungen nach dem field "category" und der image_url
        :param url:
        :return: dictionary mit den einzelnen Wetter-Kategorien als keys und einer Liste dazugehöriger Bilder als value:
                in der Form: {"category1":[Bild1,Bild,3,Bild4..], "category2":[Bild2,Bild5,..],..}

        """
        meldungen = cls._service.get_json_data(url,"meldungen")

        weather_categories_dict= {}
        for meldung in meldungen:
            if meldung.get("imageUrl") is None:
                continue
            else:
                if meldung["category"] not in weather_categories_dict:
                    weather_categories_dict[meldung["category"]] = []

                weather_categories_dict[meldung["category"]].append(meldung["imageUrl"])

        return weather_categories_dict

    def download_image(cls,url, local_path):
        """
        Herunterladen eines Bildes von einer URL und Speichern in einem lokalen Ordner.

        :param url: Die URL des Bildes, das heruntergeladen werden soll.
        :param local_path: Der Pfad und Name der Datei, in die das Bild gespeichert werden soll.
        """
        if not os.path.exists(os.path.dirname(local_path)):
            os.makedirs(os.path.dirname(local_path))

        try:
            urllib.request.urlretrieve(url, local_path)
            print("Das Bild wurde erfolgreich heruntergeladen und gespeichert.")
        except Exception as e:
            print("Fehler beim Herunterladen des Bildes: ", e)

    def downloading_files_from_a_list(cls, list_name, folder_path):

        for i in list_name:
            future_file_name = folder_path + "/" + cls.translate_into_clean_filename(i).split("processed")[1] + ".jpg"
            cls._instance.download_image(
                 i, future_file_name,)
        return

    def translate_into_clean_filename(cls, filename):
        chars_to_replace = '<>:"/\\|?*'

        # Erstellen einer Übersetzungstabelle, die die Zeichen aus chars_to_replace durch Unterstriche ersetzt
        translation_table = str.maketrans(chars_to_replace, '_' * len(chars_to_replace))

        cleaned_filename = filename.translate(translation_table)

        return cleaned_filename




