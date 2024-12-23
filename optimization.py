import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
import seaborn as sns
from lightgbm.callback import early_stopping

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result_dir = f'results_{timestamp}'
os.makedirs(result_dir, exist_ok=True)

class OptimizedLightGBM:
    def __init__(self, time_budget=3600):
        self.time_budget = time_budget
        self.start_time = None
        self.best_score = 0
        self.best_model = None
        self.cv_results = []
        
    def time_remaining(self):
        elapsed = time.time() - self.start_time
        return self.time_budget - elapsed
    
    def train_evaluate(self, X_train, y_train, X_val, y_val):
        base_params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'verbosity': -1,
        'n_jobs': -1,
        'force_row_wise': True,
        'deterministic': True,
        'feature_pre_filter': True, 
        'max_bin': 255,
        'bagging_freq': 5,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'device_type': 'gpu' if len(X_train) > 100000 else 'cpu'
    }
        
        param_grid = {
            'learning_rate': [0.01, 0.05],
            'n_estimators': [100, 200],
            'num_leaves': [31, 63],
            'min_child_samples': [20],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'reg_alpha': [0.1],
            'reg_lambda': [0.1]
        }
        
        n_splits = 3
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        train_scores = []
        val_scores = []
        feature_importance_list = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"Training fold {fold+1}/{n_splits}...")
            if self.time_remaining() < 300:
                break
                
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = LGBMClassifier(**base_params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                callbacks=[early_stopping(stopping_rounds=10)],
                eval_metric='average_precision'
            )
            
            train_pred = model.predict_proba(X_fold_train)[:, 1]
            val_pred = model.predict_proba(X_fold_val)[:, 1]
            
            train_score = average_precision_score(y_fold_train, train_pred)
            val_score = average_precision_score(y_fold_val, val_pred)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
            
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            })
            feature_importance_list.append(feature_importance)
            
            plt.figure(figsize=(10, 6))
            plt.plot(model.evals_result_['valid_0']['average_precision'], label='Validation')
            plt.title(f'Learning Curve - Fold {fold+1}')
            plt.xlabel('Iterations')
            plt.ylabel('Average Precision')
            plt.legend()
            plt.savefig(f'{result_dir}/learning_curve_fold_{fold+1}.png')
            plt.close()
            
            print(f"Fold {fold+1} - Train Score: {train_score:.4f}, Val Score: {val_score:.4f}")
            
        avg_importance = pd.concat(feature_importance_list).groupby('feature').mean()
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y=avg_importance.index, data=avg_importance.reset_index())
        plt.title('Average Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{result_dir}/feature_importance.png')
        plt.close()
        
        cv_results = pd.DataFrame({
            'fold': range(len(train_scores)),
            'train_score': train_scores,
            'val_score': val_scores
        })
        cv_results.to_csv(f'{result_dir}/cv_results.csv', index=False)
        
        return np.mean(val_scores), model
    
    def optimize(self, X_train, y_train, X_test, y_test):
        self.start_time = time.time()
        
        best_score, best_model = self.train_evaluate(X_train, y_train, X_test, y_test)
        
        test_pred = best_model.predict_proba(X_test)[:, 1]
        test_score = average_precision_score(y_test, test_pred)
        
        plt.figure(figsize=(10, 6))
        plt.hist(test_pred[y_test==0], bins=50, alpha=0.5, label='Negative Class')
        plt.hist(test_pred[y_test==1], bins=50, alpha=0.5, label='Positive Class')
        plt.title('Prediction Distribution on Test Set')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f'{result_dir}/prediction_distribution.png')
        plt.close()
        
        with open(f'{result_dir}/summary.txt', 'w') as f:
            f.write(f'Training completed in {time.time() - self.start_time:.2f} seconds\n')
            f.write(f'Best validation score: {best_score:.4f}\n')
            f.write(f'Test score: {test_score:.4f}\n')
        
        return best_model, test_score

data_dir = "data" 
X_train = pd.read_csv(f'{data_dir}/processed_X_train_v2.csv')
y_train = pd.read_csv(f'{data_dir}/processed_y_train_v2.csv')['is_fraud']
X_test = pd.read_csv(f'{data_dir}/processed_X_test_v2.csv')
y_test = pd.read_csv(f'{data_dir}/processed_y_test_v2.csv')['is_fraud']

optimizer = OptimizedLightGBM(time_budget=3600)
best_model, test_score = optimizer.optimize(X_train, y_train, 
                                          X_test, y_test)