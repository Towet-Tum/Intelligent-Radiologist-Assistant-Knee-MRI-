from IntelligentKneeMRIClassifier import logger
from IntelligentKneeMRIClassifier.config.configuration import ConfigurationManager
from IntelligentKneeMRIClassifier.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion" 
class DataIngestionPipeline:
    def __init__(self):
        pass 

    def main(self):
        config = ConfigurationManager()
        ingestion_config = config.get_data_ingestion_config()
        inges = DataIngestion(config=ingestion_config)
        inges.download_file()
        inges.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n\n >>>>>>>>>> The {STAGE_NAME} has started <<<<<<<<<<<<< ")
        ing = DataIngestionPipeline()
        ing.main()
        logger.info(f">>>>>>>>>>> The {STAGE_NAME} has compled succefully <<<<<<<<<<<<<<< \n\n")
    except Exception as e:
        logger.exception(e)
        raise e