from IntelligentKneeMRIClassifier import logger 
from IntelligentKneeMRIClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
STAGE_NAME = "Data Ingestion"
try:
    logger.info(f"\n\n\n >>>>>>>>>> The {STAGE_NAME} has started <<<<<<<<<<<<< ")
    ing = DataIngestionPipeline()
    ing.main()
    logger.info(f">>>>>>>>>>> The {STAGE_NAME} has compled succefully <<<<<<<<<<<<<<< \n\n")
except Exception as e:
    logger.exception(e)
    raise e