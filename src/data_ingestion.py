import pandas as pd
import os
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(train_path: str, test_path: str) -> tuple:
    """Load train and test data from CSV files."""
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        logger.debug('Train data loaded from %s', train_path)
        logger.debug('Test data loaded from %s', test_path)
        logger.debug('Train shape: %s', df_train.shape)
        logger.debug('Test shape: %s', df_test.shape)
        return df_train, df_test
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def save_data(df_train: pd.DataFrame, df_test: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        processed_data_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        
        df_train.to_csv(os.path.join(processed_data_path, "train.csv"), index=False)
        df_test.to_csv(os.path.join(processed_data_path, "test.csv"), index=False)
        
        logger.debug('Train and test data saved to %s', processed_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        
        train_path = params['data_ingestion']['train_path']
        test_path = params['data_ingestion']['test_path']
        
        df_train, df_test = load_data(train_path, test_path)
        
        save_data(df_train, df_test, data_path='./data')
        
        logger.info('Data ingestion completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()