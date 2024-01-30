from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path 
    local_data_file: Path 
    unzip_dir: Path 

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path 
    model_path: Path 
    epochs: int
    imgsz: int 
    batch_size: int 
    lr: float 

    axial_path: Path 
    coronal_path: Path 
    sagital_path: Path
     
    abnormal_csv_path: Path 
    acl_csv_path: Path 
    meniscus_csv_path: Path
