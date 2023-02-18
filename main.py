from cnn.Preprocessing_Service import PreprocessingService
import os

from cnn.model2_hp import WeatherCNN

if __name__ == "__main__":
    prepro_serv = PreprocessingService.instance()
    meldungen_dict = prepro_serv.filter_meldungen_after_category_and_image_url(
        "https://s3.eu-central-1.amazonaws.com/app-prod-static.warnwetter.de/v16/crowd_meldungen_overview_v2.json")

    for key, val in meldungen_dict.items():
        print(key, len(val))

    for key in meldungen_dict.keys():
        prepro_serv.downloading_files_from_a_list(
            meldungen_dict[key], f"C:/Users/marlo/PycharmProjects/WETTER/{key}")

    for label in os.listdir(r"C:/Users/marlo/PycharmProjects/WETTER"):
        prepro_serv.resize_images(f"C:/Users/marlo/PycharmProjects/WETTER/{label}", 180, 180)

    cnn = WeatherCNN(r"C:\Users\marlo\PycharmProjects\weather_dataset_from_kaggle", 11)
    cnn.search_optimal_hyperparameter()
    cnn.plot_training_process(cnn.train())
    cnn.evaluate()





