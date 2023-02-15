import unittest
from unittest.mock import MagicMock, patch
from cnn.Preprocessing_Service import PreprocessingService


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Legt die Konfigurationen (z.B Instanzen oder Variablen die bei jedem Test benötigt werden) vor einem Testdurchlauf fest"""
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
            {"category": "sunny", "imageUrl": "https://example.com/image1.jpg","unnecessaryInfo":"this should not be shown"},
            {"category": "lighting", "imageUrl": "https://example.com/image2.jpg","another unnecessaryInfo": 42},
            {"category": "sunny", "imageUrl": "https://example.com/image3.jpg"},
            {"category": "sunny", "imageUrl": None},
        ]
        expected_output = {"sunny": ["https://example.com/image1.jpg", "https://example.com/image3.jpg"],"lighting": ["https://example.com/image2.jpg"]}

        mock_service.get_json_data.return_value = meldungen

        output = self.serv.filter_meldungen_after_category_and_image_url(url)

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    unittest.main()
