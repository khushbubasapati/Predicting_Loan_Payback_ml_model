import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
import yaml
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
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


def load_model(model_path: str):
    """Load trained pipeline."""
    try:
        model_file = os.path.join(model_path, 'model.pkl')
        
        with open(model_file, 'rb') as f:
            pipeline = pickle.load(f)
        
        logger.debug('Model loaded from %s', model_file)
        return pipeline
    except FileNotFoundError as e:
        logger.error('Model file not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise


def load_data(data_path: str) -> tuple:
    """Load test data and validation data if available."""
    try:
        processed_path = os.path.join(data_path, 'processed')
        
        X_test = pd.read_csv(os.path.join(processed_path, 'X_test.csv'))
        test_ids = pd.read_csv(os.path.join(processed_path, 'test_ids.csv')).squeeze()
        
        logger.debug('Loaded X_test: %s, test_ids: %d', X_test.shape, len(test_ids))
        
        # Try to load validation data (saved during training)
        try:
            X_val = pd.read_csv(os.path.join(processed_path, 'X_val.csv'))
            y_val = pd.read_csv(os.path.join(processed_path, 'y_val.csv')).squeeze()
            logger.debug('Loaded X_val: %s, y_val: %d', X_val.shape, len(y_val))
            return X_test, test_ids, X_val, y_val
        except FileNotFoundError:
            logger.debug('Validation data not found (will skip classification report)')
            return X_test, test_ids, None, None
            
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


def calculate_optimal_threshold(pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Calculate optimal threshold using Youden's J statistic.
    """
    try:
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
        
        # Youden's J statistic: maximize (TPR - FPR)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        
        logger.debug('Optimal threshold calculated: %.6f', best_threshold)
        logger.debug('TPR at optimal threshold: %.4f', tpr[best_idx])
        logger.debug('FPR at optimal threshold: %.4f', fpr[best_idx])
        
        return float(best_threshold)
    except Exception as e:
        logger.error('Error calculating optimal threshold: %s', e)
        raise


def evaluate_on_validation(pipeline, X_val: pd.DataFrame, y_val: pd.Series, 
                          optimal_threshold: float, output_path: str) -> None:
    """
    Generate classification report on validation set.
    Based on notebook cell 40.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        # Get probabilities
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        
        # Get predictions using optimal threshold
        y_val_pred = (y_val_prob >= optimal_threshold).astype(int)
        
        # Calculate metrics
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        logger.info('='*60)
        logger.info('VALIDATION SET EVALUATION')
        logger.info('='*60)
        logger.info('Validation AUC: %.6f', val_auc)
        logger.info('Optimal Threshold: %.6f', optimal_threshold)
        logger.info('')
        logger.info('Classification Report:')
        
        # Generate classification report
        report = classification_report(y_val, y_val_pred)
        logger.info('\n%s', report)
        logger.info('='*60)
        
        # Save classification report
        report_dict = classification_report(y_val, y_val_pred, output_dict=True)
        report_file = os.path.join(output_path, 'classification_report.json')
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        logger.debug('Classification report saved: %s', report_file)
        
    except Exception as e:
        logger.error('Error evaluating on validation set: %s', e)
        raise
    # """Generate test predictions."""
    # try:
    #     os.makedirs(output_path, exist_ok=True)
        
    #     logger.debug('Generating test predictions')
        
    #     # Get probabilities
    #     test_probs = pipeline.predict_proba(X_test)[:, 1]
        
    #     # Create submission file
    #     submission = pd.DataFrame({
    #         'id': test_ids,
    #         'loan_paid_back': test_probs
    #     })
        
    #     submission_file = os.path.join(output_path, 'submission.csv')
    #     submission.to_csv(submission_file, index=False)
        
    #     logger.debug('Submission saved: %s', submission_file)
    #     logger.debug('Shape: %s', submission.shape)
    #     logger.debug('Probability range: [%.4f, %.4f]', test_probs.min(), test_probs.max())
    #     logger.debug('Mean probability: %.4f', test_probs.mean())
    # except Exception as e:
    #     logger.error('Error generating predictions: %s', e)
    #     raise


def save_metrics(model_path: str, output_path: str, optimal_threshold: float = None) -> None:
    """Load and save metrics from model_metadata.json."""
    try:
        os.makedirs(output_path, exist_ok=True)
        
        metadata_file = os.path.join(model_path, 'model_metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Add optimal threshold if available
        if optimal_threshold is not None:
            metadata['optimal_threshold'] = round(float(optimal_threshold), 6)
        
        logger.info('='*60)
        logger.info('MODEL PERFORMANCE METRICS')
        logger.info('='*60)
        logger.info('Optuna 3-Fold CV AUC : %.6f', metadata['optuna_cv_auc'])
        logger.info('Held-Out Val AUC     : %.6f', metadata['val_auc'])
        if optimal_threshold is not None:
            logger.info('Optimal Threshold    : %.6f', optimal_threshold)
        logger.info('='*60)
        
        # Save metrics report
        metrics_file = os.path.join(output_path, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.debug('Metrics saved: %s', metrics_file)
        
    except FileNotFoundError:
        logger.warning('model_metadata.json not found, skipping metrics report')
    except Exception as e:
        logger.error('Error saving metrics: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        
        pipeline = load_model(model_path='./model')
        
        X_test, test_ids, X_val, y_val = load_data(data_path='./data')
        
        # Calculate optimal threshold and evaluate on validation set
        if X_val is not None and y_val is not None:
            optimal_threshold = calculate_optimal_threshold(pipeline, X_val, y_val)
            evaluate_on_validation(pipeline, X_val, y_val, optimal_threshold, 
                                 output_path='./reports')
            
            save_metrics(model_path='./model', output_path='./reports', 
                        optimal_threshold=optimal_threshold)
        else:
            logger.warning('Validation data not available. Skipping classification report.')
            logger.warning('To enable this, save X_val and y_val during training.')
            save_metrics(model_path='./model', output_path='./reports', 
                        optimal_threshold=None)
        
        logger.info('Model evaluation completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()