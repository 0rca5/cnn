from Wetterdienst_Service import WetterdienstService
from cnn.Preprocessing_Service import PreprocessingService
from cnn.model import WetterCNN

if __name__ == "__main__":
    serv = WetterdienstService.instance()

    prepro_serv = PreprocessingService.instance()
    meldungen_dict = prepro_serv.filter_meldungen_after_category_and_image_url(
        "https://s3.eu-central-1.amazonaws.com/app-prod-static.warnwetter.de/v16/crowd_meldungen_overview_v2.json")

    for key,val in meldungen_dict.items():
         print(key, len(val))

    for key in meldungen_dict.keys():
        prepro_serv.downloading_files_from_a_list(
            meldungen_dict[key], f"C:/Users/marlo/PycharmProjects/WETTER/{key}")


