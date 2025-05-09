import file_process.data_processing as data_processing
import file_process.dataset_processing as dataset_processing
from file_process.create_spectrogram import create_spectrogram
from file_process.create_npy import create_npy
import os

input_data = '../RawDataSet/EmoDB'
processed_data = '../ProcessedDataSet/Processed'
split_data = '../ProcessedDataSet/Split'
spectrogram_file = 'features/mel_spectrogram'
npy_file = 'features/mel_npy'

current_dir = os.path.dirname(os.path.abspath(__file__))
input_data = os.path.join(current_dir, input_data)
processed_data = os.path.join(current_dir, processed_data)
split_data = os.path.join(current_dir, split_data)
npy_file = os.path.join(current_dir, npy_file)
spectrogram_file = os.path.join(current_dir, spectrogram_file)

if __name__ == '__main__':
    need_split = data_processing.process_audio_folder_with_categories(input_data, processed_data)
    print(need_split)
    splited_data = dataset_processing.datasetProcessing(need_split, split_data)
    print(splited_data)

    create_npy(splited_data,npy_file)
    # create_spectrogram(splited_data,spectrogram_file)
