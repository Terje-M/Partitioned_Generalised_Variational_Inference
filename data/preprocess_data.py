# Script which downloads and preprocesses a collection of UCI datasets.
# Flag --dir specifies the relative path at which the script will work.
# The script creates a directory called "data" under --dir and downloads
# the UCI data there. It then preprocesses the data to a format ready
# to be consumed by the models. By default, this will overwrite an existing
# "data" directory. 
#
# For each dataset, the script creates an x.npy and y.npy array containing
# the model input and outputs under data/dataset_name/.


import os
import shutil
import zipfile
import urllib.request
import argparse
import numpy as np
import pandas as pd

from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,  \
                                  StandardScaler, \
                                  OrdinalEncoder, \
                                  Binarizer

from sklearn.utils import as_float_array
from sklearn.pipeline import Pipeline

argparser = argparse.ArgumentParser()
argparser.add_argument("--dir", dest="dir", required=True, type=str)
argparser.add_argument("--no-download", dest='no_download', action="store_true")
args = argparser.parse_args()


# =============================================================================
# Dataset downloader
# =============================================================================


def download_dataset(root_dir, dataset_name, info):
    print(f'Downloading {dataset_name} dataset')
    dataset_dir = f'{root_dir}/{dataset_name}'

    if os.path.exists(os.path.join(dataset_dir, "data.npy")):
        print(f'{dataset_name} dataset already exists!')
        return
    else:
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)

        base_url = info['base_url']

        # Handle the case where the data must be unzipped
        if info['zipped']:
            # Download zip file
            url = f"{base_url}/{info['zipfile']}"
            save_location = f"{dataset_dir}/{info['zipfile']}"
            urllib.request.urlretrieve(url, save_location)

            # Unzip data
            with zipfile.ZipFile(save_location, 'r') as zip_handle:
                zip_handle.extractall(save_location[:-4])

        data = []
        for i, file_name in enumerate(info['files']):

            save_location = f'{dataset_dir}/{file_name}'

            if not info['zipped']:
                if not os.path.exists(save_location):
                    url = f"{base_url}/{file_name}"
                    urllib.request.urlretrieve(url, save_location)

                    if ('post-download' in info) and \
                            (info['post-download'][i] is not None):
                        info['post-download'][i](save_location)

            _data = None

            if ('pre-process' in info) and \
                    (info['pre-process'][i] is not None):
                _data = info['pre-process'][i](save_location)

            elif save_location[-5:] == '.xlsx':
                _data = np.array(pd.read_excel(save_location), dtype=str)

            else:
                _data = np.loadtxt(save_location,
                                   dtype=str,
                                   delimiter=info['delimiter'])

            _data = _data[1:] if info['drop-header'] else _data

            try:
                rows_with_missing = np.any(_data == '?', axis=1)
            except np.AxisError:
                rows_with_missing = np.any(np.isnan(_data), axis=1)

            print(f'{np.sum(rows_with_missing)}/{rows_with_missing.shape[0]} '
                  f'rows had missing data\n')

            if info['remove-missing']:
                _data = _data[~rows_with_missing, :]

            data.append(_data)

        data = np.concatenate(data, axis=0)
        np.save(f'{dataset_dir}/data.npy', data)


def download_datasets(root_dir, datasets):
    
    # Directory to download data to
    # data_dir = root_dir + '/data'
    dataset_dir = root_dir
    
    # if os.path.exists(dataset_dir):
    #     shutil.rmtree(dataset_dir)
    #
    # os.mkdir(data_dir)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    # Loop over datasets and download them into tmp
    for dataset_name, info in datasets.items():
        download_dataset(dataset_dir, dataset_name, info)


# =============================================================================
# Dataset processor
# =============================================================================

def process_dataset(data_dir, config):
    
    data = None
    try:
        data = np.load(f"{data_dir}/data.npy")
            
    except (FileNotFoundError, OSError):
        print("Missing this dataset! Not processed.")
        return
        
    # Set random seed and shuffle data
    np.random.seed(0)
    data = data[np.random.permutation(data.shape[0])]

    # Scaler for normalising input data
    scaler = StandardScaler()
    
    # Categorical transformer for encoding categorical classes into one hot
    categorical_transformer = OneHotEncoder(sparse=False)

    # Inputs x are all columns except output column
    mask = np.full(data.shape[1], True)
    mask[config["output_column"]] = False
    x = data[:, mask]

    # Replace "?" with np.nan.
    # x[x == "?"] = np.nan

    # Preprocessor for scaling
    preprocessor = ColumnTransformer(transformers=[
        ('num', scaler, config["numerical_features"]),
        ('cat', categorical_transformer, config["categorical_features"])
    ])

    x = preprocessor.fit_transform(x)

    y = data[:, config["output_column"]].reshape(-1, 1)
    y = config["output_generator"](y)

    np.save(data_dir + "/x.npy", x)
    np.save(data_dir + "/y.npy", y)
    
    print(f'Input  shape: {x.shape}')
    print(f'Output shape: {y.shape}')

    
def adult_preprocess_test_file(file_location):
    fhandle = open(file_location, 'r')
    content_minus_first_line = '\n'.join(fhandle.read().split('\n')[1:])
    fhandle.close()
    
    fhandle = open(file_location, 'w')
    fhandle.write(content_minus_first_line)
    fhandle.close()
    
    return


def bankruptcy_preprocess_file(file_location):
    data, meta = arff.loadarff(file_location)

    processed_data = np.zeros(shape=(len(data), len(data[0])))
    for i, data_i in enumerate(data):
        for j, val_ij in enumerate(data_i):
            processed_data[i, j] = float(val_ij)

    return processed_data


def echocardiogram_preprocess_file(file_location):
    data = np.loadtxt(file_location,
                      dtype=str,
                      delimiter=",")

    data[data == "?"] = np.nan
    valid_idx = []
    for i, x in enumerate(data):
        if x[0] != "nan" and x[1] != "nan":
            valid_idx.append(i)
            x[-1] = 1 if (float(x[0]) < 12 and x[1] == "0") else 0

    data = data[np.array(valid_idx)]

    return data


ROOT_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases'

datasets = {
    
    'abalone'        : {'base_url'       : f'{ROOT_URL}/abalone/',
                        'zipped'         : False,
                        'files'          : ['abalone.data'],
                        'delimiter'      : ',',
                        'drop-header'    : False,
                        'remove-missing' : False},
    
    'adult'          : {'base_url'       : f'{ROOT_URL}/adult/',
                        'zipped'         : False,
                        'files'          : ['adult.data', 'adult.test'],
                        'delimiter'      : ',',
                        'drop-header'    : False,
                        'remove-missing' : False,
                        'post-download'  : [None, adult_preprocess_test_file]},
    
    'mushroom'       : {'base_url'       : f'{ROOT_URL}/mushroom/',
                        'zipped'         : False,
                        'files'          : ['agaricus-lepiota.data'],
                        'delimiter'      : ',',
                        'drop-header'    : False,
                        'remove-missing' : False},
    
    'credit'         : {'base_url'       : f'{ROOT_URL}/credit-screening/',
                        'zipped'         : False,
                        'files'          : ['crx.data'],
                        'delimiter'      : ',',
                        'drop-header'    : False,
                        'remove-missing' : True},
    
    'bank'           : {'base_url'       : f'{ROOT_URL}/00222/',
                        'zipped'         : True,
                        'zipfile'        : 'bank.zip', 
                        'files'          : ['bank/bank-full.csv'],
                        'delimiter'      : ';',
                        'drop-header'    : True,
                        'remove-missing' : False},
    
    'superconductor' : {'base_url'       : f'{ROOT_URL}/00464/',
                        'zipped'         : True,
                        'zipfile'        : 'superconduct.zip', 
                        'files'          : ['superconduct/train.csv'],
                        'delimiter'      : ',',
                        'drop-header'    : True,
                        'remove-missing' : False},
    
    'protein'        : {'base_url'       : f'{ROOT_URL}/00265/',
                        'zipped'         : False,
                        'files'          : ['CASP.csv'],
                        'delimiter'      : ',',
                        'drop-header'    : True,
                        'remove-missing' : False},
    
    # 'power'          : {'base_url'       : f'{ROOT_URL}/00294/',
    #                     'zipped'         : True,
    #                     'zipfile'        : 'CCPP.zip',
    #                     'files'          : ['CCPP/CCPP/Folds5x2_pp.xlsx'],
    #                     'delimiter'      : ',',
    #                     'drop-header'    : True,
    #                     'remove-missing' : False}

    'bankruptcy'     : {'base_url'       : f'{ROOT_URL}/00365/',
                        'zipped'         : True,
                        'zipfile'        : 'data.zip',
                        'files'          : ['data/1year.arff',
                                            'data/2year.arff',
                                            'data/3year.arff',
                                            'data/4year.arff',
                                            'data/5year.arff'],
                        'delimiter'      : ',',
                        'drop-header'    : False,
                        'remove-missing' : False,
                        'pre-process'    : [bankruptcy_preprocess_file
                                            for _ in range(5)]},

    # 'parkinsons'     : {'base_url'       : f'{ROOT_URL}/00470/',
    #                     'zipped'         : True,
    #                     'zipfile'        : 'data.zip',
    #                     'files'          : ['data/1year.arff',
    #                                         'data/2year.arff',
    #                                         'data/3year.arff',
    #                                         'data/4year.arff',
    #                                         'data/5year.arff'],
    #                     'delimiter'      : ',',
    #                     'drop-header'    : False,
    #                     'remove-missing' : False,
    #                     'pre-process'    : [bankruptcy_preprocess_file
    #                                         for _ in range(5)]},

    'hepatitis'      : {'base_url'       : f'{ROOT_URL}/00503/',
                        'zipped'         : True,
                        'zipfile'        : 'HCV-Egy-Data.zip',
                        'files'          : ['HCV-Egy-Data/HCV-Egy-Data.csv'],
                        'delimiter'      : ',',
                        'drop-header'    : True,
                        'remove-missing' : False},

    'drug'           : {'base_url'       : f'{ROOT_URL}/00373/',
                        'zipped'         : False,
                        'files'          : ['drug_consumption.data'],
                        'delimiter'      : ',',
                        'drop-header'    : True,
                        'remove-missing' : False},

    'echocardiogram' : {'base_url'       : f'{ROOT_URL}/echocardiogram/',
                        'zipped'         : False,
                        'files'          : ['echocardiogram.data'],
                        'delimiter'      : ',',
                        'drop-header'    : False,
                        'remove-missing' : False,
                        'pre-process'    : [echocardiogram_preprocess_file]}
}
    
    
    
# =========================================================================
# Regression datasets
# =========================================================================


# Abalone dataset
def abalone_output_generator(y):
    oe = StandardScaler()
    return oe.fit_transform(y)

abalone_config = {
    "numerical_features"   : [1, 2, 3, 4, 5, 6, 7],
    "categorical_features" : [0],
    "folder"               : "/abalone",
    "output_column"        : 8,
    "output_generator"     : abalone_output_generator
}

# Superconductor dataset
def superconductor_output_generator(y):
    oe = StandardScaler()
    return oe.fit_transform(y)

superconductor_config = {
    "numerical_features"     : list(range(81)),
    "categorical_features"   : [],
    "folder"                 : "/superconductor",
    "output_column"          : 81,
    "output_generator"       : superconductor_output_generator
}

# Protein dataset
def protein_output_generator(y):
    oe = StandardScaler()
    return oe.fit_transform(y)

protein_config = {
    "numerical_features"     : list(range(9)),
    "categorical_features"   : [],
    "folder"                 : "/protein",
    "output_column"          : 9,
    "output_generator"       : protein_output_generator
}

# Power plant dataset
def power_output_generator(y):
    oe = StandardScaler()
    return oe.fit_transform(y)

power_config = {
    "numerical_features"     : list(range(4)),
    "categorical_features"   : [],
    "folder"                 : "/power",
    "output_column"          : 4,
    "output_generator"       : power_output_generator
}


# =========================================================================
# Classification datasets
# =========================================================================

# Adult dataset
def adult_output_generator(y):
    oe = OrdinalEncoder()
    y[y == " <=50K."] = " <=50K"
    y[y == " >50K."] = " >50K"
    return oe.fit_transform(y)

adult_config = {
    "numerical_features"   : [0, 2, 4, 10, 11, 12],
    "categorical_features" : [1, 3, 5, 6, 7, 8, 9, 13],
    "folder"               : "/adult",
    "output_column"        : 14,
    "output_generator"     : adult_output_generator
}

# Mushroom dataset
def mushroom_output_generator(y):
    oe = OrdinalEncoder()
    return oe.fit_transform(y)

mushroom_config = {
    "numerical_features"   : [],
    "categorical_features" : [i for i in range(1, 22)],
    "folder"               : "/mushroom",
    "output_column"        : 0,
    "output_generator"     : mushroom_output_generator
}

# Credit approval dataset
def credit_output_generator(y):
    oe = OrdinalEncoder()
    return oe.fit_transform(y)

credit_config = {
    "numerical_features"   : [1, 2, 7, 10, 13, 14],
    "categorical_features" : [0, 3, 4, 5, 6, 8, 9, 11, 12],
    "folder"               : "/credit",
    "output_column"        : 15,
    "output_generator"     : credit_output_generator
}

# Bank marketing dataset
def bank_output_generator(y):
    oe = OrdinalEncoder()
    return oe.fit_transform(y)

bank_config = {
    "numerical_features"   : [0, 5, 9, 11, 12, 13, 14],
    "categorical_features" : [1, 2, 3, 4, 6, 7, 8, 10, 15],
    "folder"               : "/bank",
    "output_column"        : 16,
    "output_generator"     : bank_output_generator
}


# Polish bankruptcy dataset
def bankruptcy_output_generator(y):
    oe = OrdinalEncoder()
    return oe.fit_transform(y)


bankruptcy_config = {
    "numerical_features"   : [i for i in range(64)],
    "categorical_features" : [],
    "folder"               : "/bankruptcy",
    "output_column"        : 64,
    "output_generator"     : bankruptcy_output_generator
}


# Parkinson's disease classification dataset
def parkinsons_output_generator(y):
    oe = OrdinalEncoder()
    return oe.fit_transform(y)


parkinsons_config = {
    "numerical_features"   : [i for i in range(754)],
    "categorical_features" : [],
    "folder"               : "/parkinsons",
    "output_column"        : 754,
    "output_generator"     : parkinsons_output_generator
}


# Hepatitis C dataset.
def hepatitis_output_generator(y):
    oe = OrdinalEncoder()
    return oe.fit_transform(y)


hepatitis_config = {
    "numerical_features"   : [0, 2] + list(range(10, 26)),
    "categorical_features" : [1] + list(range(3, 10)) + [27],
    "folder"               : "/hepatitis",
    "output_column"        : 28,
    "output_generator"     : hepatitis_output_generator,
}


# Drug use dataset
def drug_output_generator(y):
    oe = OrdinalEncoder()
    return oe.fit_transform(y)


drug_config = {
    "numerical_features"   : [i for i in range(1, 12)],
    "categorical_features" : [i for i in range(12, 31)],
    "folder"               : "/drug",
    "output_column"        : 31,
    "output_generator"     : drug_output_generator
}


# Echocardiogram dataset.
def echocardiogram_output_generator(y):
    oe = OrdinalEncoder()
    return oe.fit_transform(y)


echocardiogram_config = {
    "numerical_features"   : [2, 4, 5, 6, 7, 8],
    "categorical_features" : [3],
    "folder"               : "/echocardiogram",
    "output_column"        : 12,
    "output_generator"     : echocardiogram_output_generator
}

# =========================================================================
# Download and preprocess data
# =========================================================================

if not args.no_download:
    download_datasets(f"{args.dir}", datasets)


print("Processing abalone dataset")
process_dataset(f"{args.dir}/abalone", abalone_config)

print("Processing adult dataset")
process_dataset(f"{args.dir}/adult", adult_config)

print("Processing mushroom dataset")
process_dataset(f"{args.dir}/mushroom", mushroom_config)

print("Processing credit dataset")
process_dataset(f"{args.dir}/credit", credit_config)

print("Processing bank dataset")
process_dataset(f"{args.dir}/bank", bank_config)

print("Processing superconductor dataset")
process_dataset(f"{args.dir}/superconductor", superconductor_config)

print("Processing protein dataset")
process_dataset(f"{args.dir}/protein", protein_config)

# print("Processing power dataset")
# process_dataset(f"{args.dir}/power", power_config)

print("Processing bankruptcy dataset")
process_dataset(f"{args.dir}/bankruptcy", bankruptcy_config)

print("Processing parkinsons dataset")
process_dataset(f"{args.dir}/parkinsons", parkinsons_config)

print("Processing hepatitis dataset")
process_dataset(f"{args.dir}/hepatitis", hepatitis_config)

print("Processing drug dataset")
process_dataset(f"{args.dir}/drug", drug_config)

print("Processing echocardiogram dataset")
process_dataset(f"{args.dir}/echocardiogram", echocardiogram_config)
