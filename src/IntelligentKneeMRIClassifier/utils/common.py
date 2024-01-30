import os
from box.exceptions import BoxValueError
import yaml
from IntelligentKneeMRIClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import numpy as np 
import pandas as pd 
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder




@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    







# Function to preprocess .npy images
def preprocess_npy_images(file_paths, max_depth, batch_size):
    processed_images_list = []
    
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        batch_images = [np.load(file_path) for file_path in batch_paths]

        # Ensure all images in the batch have the same height and width
        if not all(image.shape[1:] == batch_images[0].shape[1:] for image in batch_images):
            print("Shapes of images in the batch:", [image.shape for image in batch_images])
            raise ValueError("All input arrays in a batch must have the same height and width.")

        # Pad or truncate the depth dimension to match the maximum depth in the batch
        processed_batch = np.stack([
            np.pad(image, ((0, max_depth - image.shape[0]), (0, 0), (0, 0)), mode='constant')
            if image.shape[0] < max_depth
            else image[:max_depth]
            for image in batch_images
        ])
        processed_images_list.append(processed_batch)
        
    return np.concatenate(processed_images_list, axis=0)



# Function to scale pixel values and encode labels
def preprocess_data(axial_files, coronal_files, sagittal_files,
                    abnormal_csv_path, acl_csv_path, meniscus_csv_path,
                      max_depth, batch_size):
    # Load CSV files for labels
    abnormal_labels = pd.read_csv(abnormal_csv_path)['1'].tolist()
    acl_labels = pd.read_csv(acl_csv_path)['0'].tolist()
    meniscus_labels = pd.read_csv(meniscus_csv_path)['0'].tolist()

    # Combine labels into a single list
    combined_labels = []
    combined_labels.extend(abnormal_labels)
    combined_labels.extend(acl_labels)
    combined_labels.extend(meniscus_labels)

    # Load and preprocess .npy images for each view
    axial_images = preprocess_npy_images(axial_files, max_depth, batch_size)
    coronal_images = preprocess_npy_images(coronal_files, max_depth, batch_size)
    sagittal_images = preprocess_npy_images(sagittal_files, max_depth, batch_size)

    # Stack images along the depth axis
    stacked_images = np.concatenate([axial_images, coronal_images, sagittal_images], axis=-1)

    # Scale the pixel values and encode labels
    scaler = StandardScaler()
    scaled_images = scaler.fit_transform(stacked_images.reshape(-1, stacked_images.shape[-1])).reshape(stacked_images.shape)

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(combined_labels)
    one_hot_encoded_labels = to_categorical(integer_labels, num_classes=3)

    # Ensure consistent number of samples for both images and labels
    min_samples = min(scaled_images.shape[0], one_hot_encoded_labels.shape[0])
    scaled_images = scaled_images[:min_samples]
    one_hot_encoded_labels = one_hot_encoded_labels[:min_samples]

    return scaled_images, one_hot_encoded_labels



