import pandas as pd
import os
import json
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
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


def load_data(data_path: str) -> tuple:
    """Load train and test data from processed directory."""
    try:
        train_path = os.path.join(data_path, 'processed', 'train.csv')
        test_path = os.path.join(data_path, 'processed', 'test.csv')
        
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        logger.debug('Loaded train: %s, test: %s', df_train.shape, df_test.shape)
        return df_train, df_test
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


def preprocess_data(df_train: pd.DataFrame, df_test: pd.DataFrame, params: dict) -> tuple:
    """
    Preprocess data: feature/target split, cast categoricals, build grade order.
    """
    try:
        target_col = params['data_ingestion']['target_column']
        id_col = params['data_ingestion']['id_column']
        cat_cols = params['features']['categorical']
        
        # Save test IDs 
        test_ids = df_test[id_col].copy()
        logger.debug('Test IDs saved')
        
        # Feature/target split 
        X_train = df_train.drop([id_col, target_col], axis=1)
        y_train = df_train[target_col]
        X_test = df_test.drop(id_col, axis=1)
        
        logger.debug('Feature/target split: X_train=%s, y_train=%s, X_test=%s', 
                    X_train.shape, y_train.shape, X_test.shape)
        
        # Cast categoricals to str
        logger.debug('Casting categorical columns to str')
        for col in cat_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype(str)
            if col in X_test.columns:
                X_test[col] = X_test[col].astype(str)
        
        # Build grade order
        grades = sorted(
            X_train['grade_subgrade'].unique(),
            key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else 0)
        )
        logger.debug('Grade order built: %d values (A1 to G5)', len(grades))
        
        return X_train, y_train, X_test, test_ids, grades
        
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, 
              test_ids: pd.Series, grades: list, data_path: str) -> None:
    """Save preprocessed data."""
    try:
        processed_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_path, exist_ok=True)
        
        X_train.to_csv(os.path.join(processed_path, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(processed_path, 'y_train.csv'), index=False, header=True)
        X_test.to_csv(os.path.join(processed_path, 'X_test.csv'), index=False)
        test_ids.to_csv(os.path.join(processed_path, 'test_ids.csv'), index=False, header=True)
        
        # Save grade order
        with open(os.path.join(processed_path, 'grade_order.json'), 'w') as f:
            json.dump(grades, f)
        
        logger.debug('Preprocessed data saved to %s', processed_path)
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        
        df_train, df_test = load_data(data_path='./data')
        
        X_train, y_train, X_test, test_ids, grades = preprocess_data(
            df_train, df_test, params
        )
        
        save_data(X_train, y_train, X_test, test_ids, grades, data_path='./data')
        
        logger.info('Data preprocessing completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()