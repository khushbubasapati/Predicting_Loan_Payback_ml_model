import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
import yaml
import optuna
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_training.log')
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
    """Load preprocessed data."""
    try:
        processed_path = os.path.join(data_path, 'processed')
        
        X_train = pd.read_csv(os.path.join(processed_path, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(processed_path, 'y_train.csv')).squeeze()
        
        with open(os.path.join(processed_path, 'grade_order.json'), 'r') as f:
            grades = json.load(f)
        
        logger.debug('Loaded X_train: %s, y_train: %s', X_train.shape, y_train.shape)
        return X_train, y_train, grades
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


def build_preprocessor(params: dict, grades: list) -> ColumnTransformer:
    """Build preprocessing pipeline (notebook cell 23)."""
    try:
        cat_cols = params['features']['categorical']
        nominal_cols = [c for c in cat_cols if c != 'grade_subgrade']
        ordinal_cols = ['grade_subgrade']
        numeric_cols = params['features']['numeric']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols),
                ('ordinal', OrdinalEncoder(categories=[grades], handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
                ('num', 'passthrough', numeric_cols),
            ],
            remainder='drop'
        )
        
        logger.debug('Preprocessor built: %d nominal, %d ordinal, %d numeric', 
                    len(nominal_cols), len(ordinal_cols), len(numeric_cols))
        return preprocessor
    except Exception as e:
        logger.error('Error building preprocessor: %s', e)
        raise


def optuna_tuning(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer,
                  scale_pos_weight: float, params: dict) -> optuna.Study:
    """Optuna hyperparameter tuning on FULL X_train with 3-fold CV (notebook cell 26)."""
    try:
        n_trials = params['model_training']['n_trials']
        cv_folds = params['model_training']['cv_folds']
        random_state = params['model_training']['random_state']
        
        def objective(trial):
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 600),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
                'max_delta_step': 1,
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'scale_pos_weight': scale_pos_weight,
                'random_state': random_state,
            }
            
            model = xgb.XGBClassifier(**model_params)
            pipeline_trial = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scores = cross_val_score(
                pipeline_trial, X_train, y_train,
                cv=cv, scoring='roc_auc', n_jobs=-1
            )
            return scores.mean()
        
        logger.debug('Starting Optuna: %d trials x %d-fold CV', n_trials, cv_folds)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.debug('Optuna complete. Best CV AUC: %.6f', study.best_value)
        return study
    except Exception as e:
        logger.error('Error during Optuna tuning: %s', e)
        raise


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer,
                      best_params: dict, scale_pos_weight: float, params: dict) -> tuple:
    """Train final model on X_tr (notebook cells 29, 31)."""
    try:
        random_state = params['model_training']['random_state']
        val_size = params['model_training']['val_size']
        
        # Split for final evaluation 
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=random_state
        )
        logger.debug('Split: X_tr=%s, X_val=%s', X_tr.shape, X_val.shape)
        
        # Build final model with best params 
        final_params = dict(best_params)
        final_params.update({
            'max_delta_step': 1,
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
        })
        
        final_model = xgb.XGBClassifier(**final_params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', final_model)
        ])
        
        # Train on X_tr 
        logger.debug('Training final pipeline on X_tr')
        pipeline.fit(X_tr, y_tr)
        logger.debug('Pipeline fitted on X_tr: %s', X_tr.shape)
        
        return pipeline, X_val, y_val
    except Exception as e:
        logger.error('Error training final model: %s', e)
        raise


def save_model(pipeline: Pipeline, study: optuna.Study, X_val: pd.DataFrame, 
               y_val: pd.Series, model_path: str, data_path: str) -> None:
    """Save model, metadata, and validation data (notebook cell 42)."""
    try:
        os.makedirs(model_path, exist_ok=True)
        
        # Save pipeline
        model_file = os.path.join(model_path, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(pipeline, f)
        logger.debug('Model saved: %s', model_file)
        
        # Compute validation AUC
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        # Save metadata
        metadata = {
            'optuna_cv_auc': round(float(study.best_value), 6),
            'val_auc': round(float(val_auc), 6),
            'best_params': {
                k: (float(v) if isinstance(v, float) else int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in study.best_params.items()
            }
        }
        
        metadata_file = os.path.join(model_path, 'model_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug('Metadata saved: %s', metadata_file)
        logger.debug('Optuna CV AUC: %.6f', metadata['optuna_cv_auc'])
        logger.debug('Validation AUC: %.6f', metadata['val_auc'])
        
        # Save validation data for use in evaluation script
        processed_path = os.path.join(data_path, 'processed')
        X_val.to_csv(os.path.join(processed_path, 'X_val.csv'), index=False)
        y_val.to_csv(os.path.join(processed_path, 'y_val.csv'), index=False, header=True)
        logger.debug('Validation data saved to %s', processed_path)
        
    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        
        X_train, y_train, grades = load_data(data_path='./data')
        
        preprocessor = build_preprocessor(params, grades)
        
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        logger.debug('scale_pos_weight: %.4f', scale_pos_weight)
        
        study = optuna_tuning(X_train, y_train, preprocessor, scale_pos_weight, params)
        
        pipeline, X_val, y_val = train_final_model(
            X_train, y_train, preprocessor,
            study.best_params, scale_pos_weight, params
        )
        
        save_model(pipeline, study, X_val, y_val, model_path='./model', data_path='./data')
        
        logger.info('Model training completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the model training process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()