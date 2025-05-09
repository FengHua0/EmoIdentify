import file_process.data_processing as data_processing
import file_process.dataset_processing as dataset_processing
import file_process.feature_extraction_1 as model_1
import file_process.feature_extraction_2 as model_2
import os

input_data = '../RawDataSet/CREMA-D'
processed_data = '../ProcessedDataSet/Processed'
split_data = '../ProcessedDataSet/Split'
features_1 = 'features/feature_extraction_1'
features_2 = 'features/feature_extraction_2'

current_dir = os.path.dirname(os.path.abspath(__file__))
input_data = os.path.join(current_dir, input_data)
processed_data = os.path.join(current_dir, processed_data)
split_data = os.path.join(current_dir, split_data)
features_1 = os.path.join(current_dir, features_1)
features_2 = os.path.join(current_dir, features_2)

if __name__ == '__main__':
    need_split = data_processing.process_audio_folder_with_categories(input_data, processed_data)
    print(need_split)
    splited_data = dataset_processing.datasetProcessing(need_split, split_data)
    print(splited_data)
    feature_1 = model_1.process_audio_folder(splited_data, features_1)
    print(feature_1)
    # feature_2 = model_2.process_audio_folder(splited_data, features_2)
    # print(feature_2)