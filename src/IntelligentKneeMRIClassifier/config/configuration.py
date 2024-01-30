import os
from IntelligentKneeMRIClassifier.constants import *
from IntelligentKneeMRIClassifier.entity.config_entity import (DataIngestionConfig, 
                                                               TrainingConfig)
from IntelligentKneeMRIClassifier.utils.common import read_yaml, create_directories

class ConfigurationManager:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training 
        params = self.params 
        
        axial_path = os.path.join("artifacts", "data_ingestion", "MRNet-v1.0", "train", "axial")
        coronal_path = os.path.join("artifacts", "data_ingestion", "MRNet-v1.0", "train" ,"coronal")
        sagital_path = os.path.join("artifacts", "data_ingestion", "MRNet-v1.0", "train", "sagittal")

        abnormal_csv_path = os.path.join("artifacts", "data_ingestion", "MRNet-v1.0", "train-abnormal.csv")
        acl_csv_path = os.path.join("artifacts", "data_ingestion", "MRNet-v1.0", "train-acl.csv")
        meniscus_csv_path = os.path.join("artifacts", "data_ingestion", "MRNet-v1.0", "train-meniscus.csv")

        create_directories([config.root_dir])
        trainng_config = TrainingConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,

            epochs=params.EPOCHS,
            batch_size=params.BATCH_SIZE,
            imgsz=params.IMG_SZ,
            lr=params.LEARNING_RATE,
            
            axial_path=Path(axial_path),
            coronal_path=Path(coronal_path),
            sagital_path=Path(sagital_path),

            abnormal_csv_path=Path(abnormal_csv_path),
            acl_csv_path=Path(acl_csv_path),
            meniscus_csv_path=Path(meniscus_csv_path)
            

        )
        return trainng_config 