"""
download_uea.py
============================
Downloads the UEA archive and converts to torch tensors. The data is saved in /processed.
"""
import os
import warnings
import numpy as np
import torch
from tqdm import tqdm
from sktime.utils.load_data import load_from_arff_to_dataframe
from sklearn.preprocessing import LabelEncoder
from helpers import download_url, unzip, mkdir_if_not_exists, save_pickle
import pandas as pd

DATA_DIR = '../data'


def download(dataset='uea'):
    """ Downloads the uea data to '/raw/uea'. """
    raw_dir = DATA_DIR + '/raw'
    assert os.path.isdir(raw_dir), "No directory exists at data/raw. Please make one to continue."

    if dataset == 'uea':
        url = 'http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip'
        save_dir = DATA_DIR + '/raw/UEA'
        zipname = save_dir + '/uea.zip'
    elif dataset == 'ucr':
        url = 'http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip'
        save_dir = DATA_DIR + '/raw/UCR'
        zipname = save_dir + '/ucr.zip'
    elif dataset == 'tsr':
        url = 'https://zenodo.org/record/3902651/files/Monash_UEA_UCR_Regression_Archive.zip?download=1'
        save_dir = DATA_DIR + '/raw/TSR'
        zipname = save_dir + '/tsr.zip'
    else:
        raise ValueError('Can only download uea, ucr or tsr. Was asked for {}.'.format(dataset))

    if os.path.exists(save_dir):
        print('Path already exists at {}. If you wish to re-download you must delete this folder.'.format(save_dir))
        return

    mkdir_if_not_exists(save_dir)

    if len(os.listdir(save_dir)) == 0:
        download_url(url, zipname)
        unzip(zipname, save_dir)


def create_torch_data(train_file, test_file):
    """Creates torch tensors for test and training from the UCR arff format.

    Args:
        train_file (str): The location of the training data arff file.
        test_file (str): The location of the testing data arff file.

    Returns:
        data_train, data_test, labels_train, labels_test: All as torch tensors.
    """
    # Suppress pandas performance warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        # Get arff format
        train_data, train_labels = load_from_arff_to_dataframe(train_file)
        test_data, test_labels = load_from_arff_to_dataframe(test_file)

    def convert_data(data):
        # Expand the series to numpy (using map instead of deprecated applymap)
        data_expand = data.map(lambda x: x.values).values
        # Single array, then to tensor
        data_numpy = np.stack([np.vstack(x).T for x in data_expand])
        tensor_data = torch.from_numpy(data_numpy).float()  # More explicit dtype
        return tensor_data

    train_data, test_data = convert_data(train_data), convert_data(test_data)

    # Encode labels as often given as strings
    encoder = LabelEncoder().fit(train_labels)
    train_labels, test_labels = encoder.transform(train_labels), encoder.transform(test_labels)
    # Use long dtype for classification labels
    train_labels = torch.from_numpy(train_labels).long()
    test_labels = torch.from_numpy(test_labels).long()

    return train_data, test_data, train_labels, test_labels


def convert_all_files(dataset='uea'):
    """ Convert all files from a given /raw/{subfolder} into torch data to be stored in /processed. """
    assert dataset in ['uea', 'ucr']
    if dataset == 'uea':
        folder = 'UEA'
        arff_folder = DATA_DIR + '/raw/{}/Multivariate_arff'.format(folder)
    elif dataset == 'ucr':
        folder = 'UCR'
        arff_folder = DATA_DIR + '/raw/{}/Univariate_arff'.format(folder)  # Fixed path for UCR

    # Check if the arff folder exists
    if not os.path.exists(arff_folder):
        print(f"ARFF folder not found at {arff_folder}. Please check the download.")
        return

    # Time for a big for loop
    dataset_dirs = [x for x in os.listdir(arff_folder) if os.path.isdir(arff_folder + '/' + x)]
    
    for ds_name in tqdm(dataset_dirs):
        # File locations
        train_file = arff_folder + '/{}/{}_TRAIN.arff'.format(ds_name, ds_name)
        test_file = arff_folder + '/{}/{}_TEST.arff'.format(ds_name, ds_name)

        # Ready save dir
        save_dir = DATA_DIR + '/processed/{}/{}'.format(folder, ds_name)

        # If files don't exist, skip.
        if any([x.split('/')[-1] not in os.listdir(arff_folder + '/{}'.format(ds_name)) for x in (train_file, test_file)]):
            if ds_name not in ['Images', 'Descriptions']:
                print('No files found for folder: {}'.format(ds_name))
            continue
        elif os.path.isdir(save_dir):
            print('Files already exist for: {}'.format(ds_name))
            continue
        else:
            try:
                train_data, test_data, train_labels, test_labels = create_torch_data(train_file, test_file)

                # Compile train and test data together
                data = torch.cat([train_data, test_data])
                labels = torch.cat([train_labels, test_labels])

                # Remove duplicates (like in your original script)
                data_numpy = data.numpy()
                unique_rows, indices, inverse_indices = np.unique(
                    data_numpy, axis=0, return_index=True, return_inverse=True
                )
                data = data[indices]
                labels = labels[indices]
                
                if len(inverse_indices) - len(indices) > 0:
                    print(f"Removed {len(inverse_indices) - len(indices)} duplicate samples from {ds_name}")

                # Save original train test indexes in case we wish to use original splits
                original_idxs = (np.arange(0, train_data.size(0)), np.arange(train_data.size(0), data.size(0)))

                # Create save directory
                mkdir_if_not_exists(save_dir)

                # Save data
                save_pickle(data, save_dir + '/data.pkl')
                save_pickle(labels, save_dir + '/labels.pkl')
                save_pickle(original_idxs, save_dir + '/original_idxs.pkl')
                
            except Exception as e:
                print(f"Error processing {ds_name}: {str(e)}")
                continue


if __name__ == '__main__':
    dataset = 'uea'
    download(dataset)
    convert_all_files(dataset)