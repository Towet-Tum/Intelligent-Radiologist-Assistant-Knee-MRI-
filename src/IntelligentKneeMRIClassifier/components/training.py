import os 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from IntelligentKneeMRIClassifier import logger
from sklearn.model_selection import train_test_split 
from IntelligentKneeMRIClassifier.utils.common import preprocess_data
from IntelligentKneeMRIClassifier.entity.config_entity import TrainingConfig
from IntelligentKneeMRIClassifier.models.model_architecture import Knee_mri_3D_model 

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config 
    
    def dataset_split(self):
        try:
            logger.info("\n\n <<<<<<<<<<<<<<<<< The dataset splitting has started >>>>>>>>>>>> \n\n")
            # Assuming you have lists of file paths for each view
            axial_files = [os.path.join(self.config.axial_path, file) for file in os.listdir(self.config.axial_path)]
            coronal_files = [os.path.join(self.config.coronal_path, file) for file in os.listdir(self.config.coronal_path)]
            sagittal_files = [os.path.join(self.config.sagital_path, file) for file in os.listdir(self.config.sagital_path)]

            # Define depth parameters for the entire dataset
            max_depth = 44 

            batch_size = self.config.batch_size  
            # Load and preprocess data
            scaled_images, one_hot_encoded_labels = preprocess_data(axial_files, coronal_files, 
                                                                    sagittal_files,
                                                                    self.config.abnormal_csv_path,
                                                                    self.config.acl_csv_path,
                                                                    self.config.meniscus_csv_path,
                                                                      max_depth, batch_size)

            # Split the data into training and validation sets
            train_images, valid_images, train_labels, valid_labels = train_test_split(
            scaled_images, one_hot_encoded_labels, test_size=0.2, random_state=42
            )
            # Reshape the input data to include the batch_size dimension
            train_images = train_images.reshape((-1, max_depth, 256, 256, 3))
            valid_images = valid_images.reshape((-1, max_depth, 256, 256, 3))
            logger.info("<<<<<<<<<<<<<<<<< The dataset splitting has completed succefully >>>>>>>>>>>> \n\n")
        except Exception as e:
            raise e
        return train_images, valid_images, train_labels, valid_labels, max_depth
    
    def trainer(self):
        train_images, valid_images, train_labels, valid_labels, max_depth = self.dataset_split()
        model = Knee_mri_3D_model(self.max_depth, self.config.imgsz, self.config.imgsz,3, 3)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.lr), 
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), 
                  epochs=self.config.epochs, batch_size=self.config.batch_size)
        model.save(self.config.model_path)