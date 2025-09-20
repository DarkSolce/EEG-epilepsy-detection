"""
EEG Epilepsy Detection Package
=============================

This package provides a comprehensive pipeline for EEG-based epilepsy detection
with focus on interpretability and clinical performance.

Modules:
--------
- data_preprocessing: Data loading, cleaning, and preprocessing utilities
- feature_engineering: Feature selection and dimensionality reduction
- modeling: Machine learning models and evaluation for epilepsy detection

Author: Skander Chebbi
Date: 2025
Version: 1.0.0
"""

import logging
import warnings
from pathlib import Path

# Supprimer les warnings non critiques pour une sortie propre
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration du logging pour le package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Version du package
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Informations sur le package
PACKAGE_INFO = {
    'name': 'EEG Epilepsy Detection',
    'version': __version__,
    'description': 'ML Pipeline for EEG-based epilepsy detection',
    'author': __author__,
    'license': 'MIT',
    'python_requires': '>=3.8',
    'main_modules': ['data_preprocessing', 'feature_engineering', 'modeling']
}

# Créer la structure de répertoires si elle n'existe pas
def create_project_structure():
    """Créer la structure de répertoires du projet"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/interim',
        'models',
        'results',
        'reports/figures',
        'logs'
    ]
    
    project_root = Path(__file__).parent.parent
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Créer des fichiers .gitkeep pour les dossiers vides
    gitkeep_dirs = ['data/raw', 'data/processed', 'data/interim', 'models', 'results', 'logs']
    for directory in gitkeep_dirs:
        gitkeep_path = project_root / directory / '.gitkeep'
        if not gitkeep_path.exists():
            gitkeep_path.touch()

# Créer la structure au moment de l'import
create_project_structure()

# Imports des modules principaux avec gestion d'erreur
try:
    from . import data_preprocessing
    from . import feature_engineering  
    from . import modeling
    
    # Imports des classes principales pour faciliter l'utilisation
    from .data_preprocessing import EEGDataLoader, EEGPreprocessor, create_preprocessing_pipeline
    from .feature_engineering import FeatureSelector, EEGFeatureEngineer, create_feature_engineering_pipeline
    from .modeling import EEGModelTrainer, ModelEvaluator, create_complete_modeling_pipeline
    
    MODULES_LOADED = True
    
except ImportError as e:
    logging.warning(f"Certains modules n'ont pas pu être importés: {e}")
    MODULES_LOADED = False

# Configuration par défaut du projet
DEFAULT_CONFIG = {
    'data': {
        'raw_data_path': 'data/raw/EEG_Scaled_data.csv',
        'processed_data_dir': 'data/processed',
        'target_column': None  # Auto-détection
    },
    'preprocessing': {
        'test_size': 0.2,
        'random_state': 42,
        'handle_missing': 'median',
        'remove_constant_features': True,
        'standardize': True
    },
    'feature_engineering': {
        'selection_method': 'combined',
        'max_features': 2000,
        'apply_pca': False,
        'pca_variance_threshold': 0.95
    },
    'modeling': {
        'models': ['Random Forest', 'Logistic Regression', 'Gradient Boosting'],
        'handle_imbalance': True,
        'cv_folds': 5,
        'scoring_metric': 'f1',
        'quick_training': False
    },
    'output': {
        'models_dir': 'models',
        'results_dir': 'results',
        'save_artifacts': True
    }
}

def get_package_info():
    """Retourner les informations sur le package"""
    return PACKAGE_INFO.copy()

def print_welcome():
    """Afficher le message de bienvenue"""
    print("="*70)
    print(f"🧠 EEG EPILEPSY DETECTION PACKAGE v{__version__}")
    print("="*70)
    print("📊 Pipeline ML pour la détection d'épilepsie basée sur l'EEG")
    print("🎯 Focus: Interprétabilité + Performance clinique")
    print()
    print("📦 Modules disponibles:")
    if MODULES_LOADED:
        print("  ✅ data_preprocessing - Prétraitement des données EEG")
        print("  ✅ feature_engineering - Sélection et ingénierie des features") 
        print("  ✅ modeling - Modélisation ML interprétable")
    else:
        print("  ⚠️  Certains modules ne sont pas disponibles")
    print()
    print("🚀 Pour commencer:")
    print("  from src import create_complete_pipeline")
    print("  results = create_complete_pipeline('data/raw/your_data.csv')")
    print("="*70)

def create_complete_pipeline(data_path: str, config: dict = None) -> dict:
    """
    Pipeline complet de A à Z pour la détection d'épilepsie
    
    Parameters:
    -----------
    data_path : str
        Chemin vers les données EEG brutes
    config : dict, optional
        Configuration personnalisée (utilise DEFAULT_CONFIG si None)
        
    Returns:
    --------
    dict: Résultats complets du pipeline
    """
    if not MODULES_LOADED:
        raise ImportError("Modules requis non disponibles. Vérifiez l'installation.")
    
    # Configuration
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 DÉBUT DU PIPELINE COMPLET DE DÉTECTION D'ÉPILEPSIE")
    logger.info(f"Données: {data_path}")
    
    try:
        # 1. PREPROCESSING
        logger.info("📊 Phase 1: Preprocessing des données")
        preprocessing_report = create_preprocessing_pipeline(
            data_path=data_path,
            output_dir=config['data']['processed_data_dir']
        )
        
        # Charger les données preprocessées
        import numpy as np
        processed_data = np.load(f"{config['data']['processed_data_dir']}/preprocessed_data.npz")
        X_train = processed_data['X_train']
        X_test = processed_data['X_test'] 
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        
        logger.info(f"✅ Preprocessing terminé - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 2. FEATURE ENGINEERING
        logger.info("🔧 Phase 2: Ingénierie des features")
        
        # Convertir en DataFrame pour feature engineering
        import pandas as pd
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        X_train_eng, X_test_eng, feature_report = create_feature_engineering_pipeline(
            X_train=X_train_df,
            y_train=pd.Series(y_train),
            X_test=X_test_df,
            method=config['feature_engineering']['selection_method'],
            output_dir=config['data']['processed_data_dir']
        )
        
        logger.info(f"✅ Feature engineering terminé - Features: {X_train_eng.shape[1]}")
        
        # 3. MODELING
        logger.info("🤖 Phase 3: Modélisation ML")
        modeling_results = create_complete_modeling_pipeline(
            X_train=X_train_eng.values,
            y_train=y_train,
            X_test=X_test_eng.values, 
            y_test=y_test,
            model_names=config['modeling']['models'],
            quick_training=config['modeling']['quick_training'],
            output_dir=config['output']['models_dir']
        )
        
        logger.info(f"✅ Modélisation terminée - Meilleur: {modeling_results['best_model']}")
        
        # 4. RÉSULTATS FINAUX
        complete_results = {
            'pipeline_info': {
                'data_path': data_path,
                'config_used': config,
                'completion_time': logger.handlers[0].formatter.formatTime(
                    logging.LogRecord('', 0, '', 0, '', (), None), '%Y-%m-%d %H:%M:%S'
                ),
                'success': True
            },
            'preprocessing_report': preprocessing_report,
            'feature_engineering_report': feature_report,
            'modeling_results': modeling_results,
            'final_performance': {
                'best_model': modeling_results['best_model'],
                'best_metrics': modeling_results['evaluation_results'][modeling_results['best_model']]['clinical_metrics'],
                'model_comparison': modeling_results['model_comparison']
            }
        }
        
        # Sauvegarder les résultats complets
        import json
        results_path = f"{config['output']['results_dir']}/complete_pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        logger.info("🎉 PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
        logger.info(f"📊 Résultats sauvegardés: {results_path}")
        
        return complete_results
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {str(e)}")
        return {
            'pipeline_info': {
                'success': False,
                'error': str(e),
                'data_path': data_path
            }
        }

def quick_start(data_path: str) -> dict:
    """
    Démarrage rapide avec configuration par défaut
    
    Parameters:
    -----------
    data_path : str
        Chemin vers les données EEG
        
    Returns:
    --------
    dict: Résultats du pipeline
    """
    print("🚀 Démarrage rapide du pipeline EEG")
    
    # Configuration optimisée pour rapidité
    quick_config = DEFAULT_CONFIG.copy()
    quick_config['modeling']['quick_training'] = True
    quick_config['modeling']['models'] = ['Random Forest', 'Logistic Regression']
    quick_config['feature_engineering']['max_features'] = 1000
    
    return create_complete_pipeline(data_path, quick_config)

def load_model(model_name: str, models_dir: str = 'models'):
    """
    Charger un modèle sauvegardé
    
    Parameters:
    -----------
    model_name : str
        Nom du modèle (ex: 'random_forest')
    models_dir : str
        Répertoire des modèles
        
    Returns:
    --------
    sklearn model: Modèle chargé
    """
    import joblib
    import os
    
    model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    model_path = os.path.join(models_dir, model_filename)
    
    if not os.path.exists(model_path):
        available_models = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}\nModèles disponibles: {available_models}")
    
    model = joblib.load(model_path)
    logging.getLogger(__name__).info(f"Modèle chargé: {model_path}")
    
    return model

def predict_epilepsy(model, X_new, threshold: float = 0.5):
    """
    Prédire l'épilepsie sur de nouvelles données
    
    Parameters:
    -----------
    model : sklearn model
        Modèle entraîné
    X_new : array-like
        Nouvelles données EEG
    threshold : float
        Seuil de classification
        
    Returns:
    --------
    dict: Prédictions et probabilités
    """
    import numpy as np
    
    # Prédictions
    try:
        probabilities = model.predict_proba(X_new)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    except:
        predictions = model.predict(X_new)
        probabilities = None
    
    results = {
        'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
        'n_samples': len(predictions),
        'epileptic_detected': int(np.sum(predictions)),
        'percentage_epileptic': float(np.mean(predictions) * 100)
    }
    
    if probabilities is not None:
        results['probabilities'] = probabilities.tolist()
        results['confidence_scores'] = probabilities.tolist()
    
    return results

# Afficher le message de bienvenue lors de l'import
if __name__ != '__main__':
    print_welcome()

# Exports principaux du package
__all__ = [
    # Informations du package
    '__version__', 'get_package_info', 'PACKAGE_INFO', 'DEFAULT_CONFIG',
    
    # Classes principales
    'EEGDataLoader', 'EEGPreprocessor', 
    'FeatureSelector', 'EEGFeatureEngineer',
    'EEGModelTrainer', 'ModelEvaluator',
    
    # Pipelines
    'create_preprocessing_pipeline',
    'create_feature_engineering_pipeline', 
    'create_complete_modeling_pipeline',
    'create_complete_pipeline',
    
    # Utilitaires
    'quick_start', 'load_model', 'predict_epilepsy',
    'print_welcome', 'create_project_structure'
]