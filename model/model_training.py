import pandas as pd
import joblib
import numpy as np
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RobustModelTrainer:
    def __init__(self, features_file: str, labels_file: str, output_dir: str = 'model_results'):
        self.features_file = features_file
        self.labels_file = labels_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.features = None
        self.labels = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.class_weights = None

        self._load_and_prepare_data()

        self._configure_models()

    def _load_and_prepare_data(self):
        try:
            self.features = joblib.load(self.features_file)
            df_labels = pd.read_csv(self.labels_file)

            label_col = self._validate_label_column(df_labels)
            self.labels = df_labels[label_col]

            if len(self.features) != len(self.labels):
                raise ValueError(f"Feature count ({len(self.features)}) doesn't match label count ({len(self.labels)})")

            self._analyze_class_distribution()

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.features, self.labels,
                test_size=0.3,
                random_state=42,
                stratify=self.labels
            )

            logger.info("Data loaded and prepared successfully")

        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def _validate_label_column(self, df: pd.DataFrame) -> str:
        possible_labels = ['label', 'sentiment', 'target', 'class', 'category']
        for col in possible_labels:
            if col in df.columns:
                logger.info(f"Using label column: '{col}'")
                return col
        available = df.columns.tolist()
        raise ValueError(f"No recognized label column found. Available columns: {available}")

    def _analyze_class_distribution(self):
        class_counts = Counter(self.labels)
        logger.info("\nClass Distribution:")
        for cls, count in class_counts.items():
            logger.info(f"Class {cls}: {count} samples ({count / len(self.labels):.2%})")

        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        self.class_weights = dict(enumerate(self.class_weights))

    def _configure_models(self):
        self.models = {
            'logistic_regression': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('model', LogisticRegression(max_iter=1000))
                ]),
                'params': {
                    'model__C': [0.01, 0.1, 1, 10],
                    'model__penalty': ['l2'],  # Removed problematic 'none' option
                    'model__class_weight': [None, 'balanced']
                }
            },
            'svm': {
                'pipeline': Pipeline([
                    ('scaler', StandardScaler(with_mean=False)),
                    ('model', SVC(probability=True))
                ]),
                'params': {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf'],
                    'model__gamma': ['scale', 'auto'],
                    'model__class_weight': [None, 'balanced']
                }
            },
            'random_forest': {
                'pipeline': Pipeline([
                    ('model', RandomForestClassifier())
                ]),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5],
                    'model__class_weight': [None, 'balanced']
                }
            }
        }

        try:
            from xgboost import XGBClassifier
            self.models['xgboost'] = {
                'pipeline': Pipeline([
                    ('model', XGBClassifier(eval_metric='mlogloss', use_label_encoder=False))
                ]),
                'params': {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [3, 6],
                    'model__learning_rate': [0.01, 0.1],
                    'model__subsample': [0.8, 1.0]
                }
            }
        except ImportError:
            logger.warning("Error with xgboost")

    def _plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        plot_path = self.output_dir / f'{model_name}_confusion_matrix.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {plot_path}")

    def train_models(self):
        results = []

        for model_name, config in self.models.items():
            try:
                logger.info(f"\n{'=' * 60}\nTraining {model_name.replace('_', ' ').title()} Model\n{'=' * 60}")

                start_time = time.time()

                grid_search = GridSearchCV(
                    estimator=config['pipeline'],
                    param_grid=config['params'],
                    cv=5,
                    n_jobs=-1,
                    scoring='f1_weighted',
                    verbose=1,
                    error_score='raise'
                )

                grid_search.fit(self.X_train, self.y_train)

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(self.X_test)

                report = classification_report(self.y_test, y_pred, output_dict=True)
                logger.info(f"\nBest Parameters: {grid_search.best_params_}")
                logger.info(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")

                self._plot_confusion_matrix(self.y_test, y_pred, model_name)

                if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                    self._plot_feature_importance(best_model, model_name)

                model_path = self.output_dir / f'{model_name}_model.pkl'
                joblib.dump(best_model, model_path)

                results.append({
                    'model': model_name,
                    'best_params': grid_search.best_params_,
                    'accuracy': report['accuracy'],
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1_score': report['weighted avg']['f1-score'],
                    'training_time': time.time() - start_time,
                    'model_path': str(model_path)
                })

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue

        self._save_and_display_results(results)

    def _plot_feature_importance(self, model, model_name: str):
        try:
            importances = model.named_steps['model'].feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title(f"Feature Importance - {model_name.replace('_', ' ').title()}")
            plt.bar(range(20), importances[indices][:20], align='center')
            plt.xticks(range(20), indices[:20], rotation=90)
            plt.xlim([-1, 20])
            plt.tight_layout()

            plot_path = self.output_dir / f'{model_name}_feature_importance.png'
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved feature importance to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {str(e)}")

    def _save_and_display_results(self, results):
        results_df = pd.DataFrame(results)

        results_path = self.output_dir / 'model_results.csv'
        results_df.to_csv(results_path, index=False)

        logger.info("\nModel Training Summary:")
        summary_df = results_df[['model', 'accuracy', 'f1_score', 'training_time']]
        summary_df.columns = ['Model', 'Accuracy', 'F1 Score', 'Training Time (s)']

        try:
            from tabulate import tabulate
            logger.info("\n" + tabulate(summary_df, headers='keys', tablefmt='psql', showindex=False))
        except ImportError:
            logger.info("\n" + summary_df.to_string(index=False))

        if not results_df.empty:
            best_model = results_df.loc[results_df['f1_score'].idxmax()]
            logger.info(f"\nBest performing model: {best_model['model']} with F1-score: {best_model['f1_score']:.4f}")
        else:
            logger.warning("No models were trained")


if __name__ == "__main__":
    try:
        FEATURES_PATH = 'data/processed/extracted_features.pkl'
        LABELS_PATH = 'data/processed/clean_reddit_data.csv'

        trainer = RobustModelTrainer(
            features_file=FEATURES_PATH,
            labels_file=LABELS_PATH,
            output_dir='model_results'
        )
        trainer.train_models()

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise