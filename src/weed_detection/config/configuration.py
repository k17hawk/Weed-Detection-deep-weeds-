import os
from pathlib import Path
from ensure import ensure_annotations
from weed_detection import logger
from weed_detection.constants.constant import DATA_SOURCE, CONFIG_FILE_PATH, PARAMS_FILE_PATH
from weed_detection.utils.utility import read_yaml, create_directories
from weed_detection.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        
        logger.info(f"📂 Loading config from: {config_filepath}")
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
       
        
        logger.info(f"📂 Creating artifacts root: {self.config.artifacts_root}")
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info("🔍 Getting data ingestion configuration...")
    
        config = self.config.data_ingestion
        
        logger.info(f"📁 Creating root directory: {config.root_dir}")
        create_directories([config.root_dir])

        
        # Get actual URL from constants
        source_url = DATA_SOURCE
        
        logger.info(f"✅ Found URL for key: {source_url}")
        logger.info(f"📦 Dataset URL: {source_url[:50]}...")  

        # Create config object
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=source_url,  
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir) 
        )
        
        logger.info("✅ Data ingestion configuration created successfully")
        
        return data_ingestion_config


