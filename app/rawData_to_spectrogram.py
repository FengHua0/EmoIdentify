import file_process.data_processing as data_processing
import file_process.dataset_processing as dataset_processing
from app.file_process.create_spectrogram import create_spectrogram

input_data = '../RawDataSet/CREMA-D'
processed_data = '../ProcessedDataSet/Processed'
split_data = '../ProcessedDataSet/Split'
spectrogram_file = 'features/spectrogram'

if __name__ == '__main__':
    need_split = data_processing.process_audio_folder_with_categories(input_data, processed_data)
    print(need_split)
    splited_data = dataset_processing.datasetProcessing(need_split, split_data)
    print(splited_data)
    create_spectrogram(splited_data,spectrogram_file)
