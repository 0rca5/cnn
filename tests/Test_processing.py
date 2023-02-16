import unittest
from unittest.mock import MagicMock, patch
from cnn.Preprocessing_Service import PreprocessingService
import os
from PIL import Image
import tempfile


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Legt die Konfigurationen (z.B. Instanzen oder Variablen, die bei jedem Test benötigt werden) vor einem Testdurchlauf fest"""
        cls.serv = PreprocessingService.instance()
    @patch('cnn.Preprocessing_Service.PreprocessingService._service')
    def test_filter_meldungen_calls_get_json_data_with_correct_arguments(self, mock_service):
        # Arrange
        url = "http://example.com/api"
        mock_service.get_json_data.return_value = []

        # Act
        self.serv.filter_meldungen_after_category_and_image_url(url)

        # Assert
        mock_service.get_json_data.assert_called_once_with(url, "meldungen")

    @patch('cnn.Preprocessing_Service.PreprocessingService._service')
    def test_filter_meldungen_filters_by_category_and_image_url_returns_correct_filtered_dictionary(self, mock_service):
        url = "http://example.com/api"
        meldungen = [
            {"category": "sunny", "imageUrl": "https://example.com/image1.jpg","unnecessaryInfo": "this should not be shown"},
            {"category": "lighting", "imageUrl": "https://example.com/image2.jpg","another unnecessaryInfo": 42},
            {"category": "sunny", "imageUrl": "https://example.com/image3.jpg"},
            {"category": "sunny", "imageUrl": None},
        ]
        expected_output = {"sunny": ["https://example.com/image1.jpg", "https://example.com/image3.jpg"],"lighting": ["https://example.com/image2.jpg"]}

        mock_service.get_json_data.return_value = meldungen

        output = self.serv.filter_meldungen_after_category_and_image_url(url)

        self.assertEqual(output, expected_output)

    @patch('cnn.Preprocessing_Service.PreprocessingService._PreprocessingService__download_image')
    @patch('cnn.Preprocessing_Service.PreprocessingService._PreprocessingService__translate_into_clean_filename')
    def test_downloading_files_from_a_list_calls_download_image_with_correct_arguments(self,
                                                                                       mock_translate_into_clean_filename,
                                                                                       mock_download_image):

        mock_translate_into_clean_filename.return_value = "clean_filename"
        list_name = ["test.jpg"]
        folder_path = "/path/to/folder"

        self.serv.downloading_files_from_a_list(list_name, folder_path)

        expected_file_name = folder_path + "/clean_filename.jpg"
        mock_download_image.assert_called_with(list_name[0], expected_file_name)

    @patch('urllib.request.urlretrieve')
    def test_download_image_raises_exception_when_url_is_invalid(self, mock_urlretrieve):
        url = "http://invalid_url.com/image.jpg"
        local_path = "/path/to/image.jpg"
        mock_urlretrieve.side_effect = Exception("Invalid URL")

        with self.assertRaises(Exception):
            self.serv._PreprocessingService__download_image(url, local_path)

    @patch('urllib.request.urlretrieve')
    def test_download_image_calls_urlretrieve_with_correct_arguments(self, mock_urlretrieve):
        url = "http://valid_url.com/image.jpg"
        local_path = "/path/to/image.jpg"

        self.serv._PreprocessingService__download_image(url, local_path)

        mock_urlretrieve.assert_called_with(url, local_path)

    def test_translate_into_clean_filename_returns_clean_filename(self):
        filename = "<this>is/|clean?*"
        expected_clean_filename = "_this_is__clean__"

        clean_filename = self.serv._PreprocessingService__translate_into_clean_filename(filename)

        self.assertEqual(expected_clean_filename, clean_filename)

    def test_resize_images_resizes_jpg_files(self):
        # Erstellen von einem temporären Directory
        test_dir = tempfile.TemporaryDirectory()

        try:
            img_width, img_height = 200, 200

            # Erstellen von Testbildern
            test_images = []
            for i in range(3):
                img_path = os.path.join(test_dir.name, f"test{i}.jpg")
                img = Image.new("RGB", (500, 500), color=(255, 0, 0))
                with open(img_path, "wb") as f:
                    img.save(img_path)
                test_images.append((img_path, img.size))

            self.serv.resize_images(test_dir.name, img_height, img_width)

            # Überprüfung, ob die Größe jedes Bildes korrekt geändert wurde
            for img_path, original_size in test_images:
                with open(img_path, 'rb') as f:
                    img = Image.open(f)
                    self.assertEqual((img_width, img_height), img.size)
        finally:
            # Löschen des temporären Directory
            test_dir.cleanup()

    def test_resize_images_checks_format(self):
        # Erstellen von einem temporären Directory
        test_dir = tempfile.TemporaryDirectory()

        try:
            img_width, img_height = 200, 200

            # Erstellen von Testbildern
            test_images = []
            for i in range(3):
                img_path = os.path.join(test_dir.name, f"test{i}.png")
                img = Image.new("RGBA", (500, 500), color=(255, 0, 0))
                with open(img_path, "wb") as f:
                    img.save(img_path)
                test_images.append((img_path, img.size))

            self.serv.resize_images(test_dir.name, img_height, img_width)

            # Überprüfung, ob jedes Bild im RGB-Modus gespeichert wurde
            for img_path, original_size in test_images:
                with open(img_path, 'rb') as f:
                    img = Image.open(f)
                    self.assertEqual("RGB", img.mode)

        finally:
            # Löschen des temporären Directory
            test_dir.cleanup()



if __name__ == "__main__":
    unittest.main()
