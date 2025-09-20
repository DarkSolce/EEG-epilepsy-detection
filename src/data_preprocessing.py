"""
Data Preprocessing Module for EEG Epilepsy Detection
====================================================

This module provides utilities for loading, preprocessing, and preparing
EEG data for machine learning models focused on epilepsy detection.

Author: Skander Chebbi
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGDataLoader:
    """
    Classe pour charger les données EEG depuis différents formats
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.edf', '.mat']
        
    def load_csv_data(self, file_path: str, target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charger un dataset EEG depuis un fichier CSV
        
        Parameters:
        -----------
        file_path : str
            Chemin vers le fichier CSV
        target_col : str, optional
            Nom de la colonne cible. Si None, prend la dernière colonne
            
        Returns:
        --------
        X : pd.DataFrame
            Features EEG
        y : pd.Series
            Labels (0=non-épileptique, 1=épileptique)
        """
        logger.info(f"Chargement des données depuis {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
            
        # Charger le dataset
        df = pd.read_csv(file_path)
        logger.info(f"Dataset chargé: {df.shape}")
        
        # Identifier la colonne cible
        if target_col is None:
            possible_targets = ['y', 'target', 'label', 'class', 'seizure']
            target_col = None
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
            if target_col is None:
                target_col = df.columns[-1]
                logger.warning(f"Colonne cible non spécifiée, utilisation de: {target_col}")
        
        # Séparer features et target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Normaliser les labels (0=non-épileptique, 1=épileptique)
        unique_values = set(y.unique())
        if unique_values == {1, 2}:
            y = (y == 2).astype(int)
            logger.info("Labels convertis: 1→0 (normal), 2→1 (épileptique)")
        elif unique_values == {4, 5}:
            y = (y == 5).astype(int)
            logger.info("Labels convertis: 4→0 (normal), 5→1 (épileptique)")
        elif unique_values != {0, 1}:
            logger.warning(f"Valeurs de labels inattendues: {unique_values}")
            
        logger.info(f"Features: {X.shape[1]:,}, Échantillons: {X.shape[0]:,}")
        logger.info(f"Distribution: {dict(pd.Series(y).value_counts())}")
        
        return X, y
    
    def validate_data_quality(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Valider la qualité des données
        
        Returns:
        --------
        dict: Rapport de qualité des données
        """
        quality_report = {}
        
        # Valeurs manquantes
        missing_count = X.isnull().sum().sum()
        missing_features = X.isnull().sum()[X.isnull().sum() > 0]
        
        quality_report['missing_values'] = {
            'total_missing': int(missing_count),
            'features_with_missing': len(missing_features),
            'missing_percentage': float(missing_count / (X.shape[0] * X.shape[1]) * 100)
        }
        
        # Distribution des classes
        class_distribution = dict(pd.Series(y).value_counts())
        quality_report['class_distribution'] = class_distribution
        quality_report['class_imbalance_ratio'] = float(
            class_distribution.get(0, 0) / max(class_distribution.get(1, 1), 1)
        )
        
        # Statistiques des features
        quality_report['feature_stats'] = {
            'n_features': int(X.shape[1]),
            'n_samples': int(X.shape[0]),
            'features_constant': int((X.nunique() == 1).sum()),
            'features_near_constant': int((X.nunique() <= 2).sum())
        }
        
        return quality_report


class EEGPreprocessor:
    """
    Classe pour le prétraitement des données EEG
    """
    
    def __init__(self):
        self.scaler = None
        self.selected_features = None
        self.feature_selector = None
        
    def remove_constant_features(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Supprimer les features quasi-constantes
        
        Parameters:
        -----------
        X : pd.DataFrame
            Dataset des features
        threshold : float
            Seuil de variance minimale
            
        Returns:
        --------
        pd.DataFrame: Dataset sans features constantes
        """
        logger.info("Suppression des features quasi-constantes...")
        
        # Calculer la variance de chaque feature
        variances = X.var()
        constant_features = variances[variances < threshold].index
        
        if len(constant_features) > 0:
            logger.info(f"Suppression de {len(constant_features)} features quasi-constantes")
            X_filtered = X.drop(columns=constant_features)
        else:
            X_filtered = X.copy()
            
        return X_filtered
    
    def handle_missing_values(self, X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Gérer les valeurs manquantes
        
        Parameters:
        -----------
        X : pd.DataFrame
            Dataset avec valeurs manquantes potentielles
        strategy : str
            Stratégie d'imputation ('mean', 'median', 'drop')
            
        Returns:
        --------
        pd.DataFrame: Dataset sans valeurs manquantes
        """
        missing_count = X.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("Aucune valeur manquante détectée")
            return X.copy()
            
        logger.info(f"Traitement de {missing_count} valeurs manquantes avec stratégie: {strategy}")
        
        if strategy == 'drop':
            # Supprimer les lignes avec valeurs manquantes
            X_clean = X.dropna()
            logger.info(f"Lignes supprimées: {len(X) - len(X_clean)}")
        elif strategy in ['mean', 'median']:
            # Imputation
            X_clean = X.copy()
            if strategy == 'mean':
                X_clean = X_clean.fillna(X_clean.mean())
            else:
                X_clean = X_clean.fillna(X_clean.median())
            logger.info(f"Imputation {strategy} effectuée")
        else:
            raise ValueError(f"Stratégie non supportée: {strategy}")
            
        return X_clean
    
    def standardize_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple:
        """
        Standardiser les features (fit sur train, transform sur train et test)
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Dataset d'entraînement
        X_test : pd.DataFrame, optional
            Dataset de test
            
        Returns:
        --------
        tuple: (X_train_scaled, X_test_scaled) ou X_train_scaled si pas de test
        """
        logger.info("Standardisation des features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        else:
            return X_train_scaled
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                              test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Créer un split stratifié train/test
        
        Returns:
        --------
        tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Création du split train/test (test_size={test_size})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train: {len(X_train):,} échantillons")
        logger.info(f"Test: {len(X_test):,} échantillons")
        logger.info(f"Distribution train: {dict(pd.Series(y_train).value_counts())}")
        logger.info(f"Distribution test: {dict(pd.Series(y_test).value_counts())}")
        
        return X_train, X_test, y_train, y_test
        
    def save_preprocessing_artifacts(self, output_dir: str = 'data/processed') -> None:
        """
        Sauvegarder les objets de preprocessing
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler sauvegardé: {scaler_path}")
            
        if self.selected_features is not None:
            features_path = os.path.join(output_dir, 'selected_features.pkl')
            joblib.dump(self.selected_features, features_path)
            logger.info(f"Features sélectionnées sauvegardées: {features_path}")
    
    def load_preprocessing_artifacts(self, input_dir: str = 'Data/processed') -> None:
        """
        Charger les objets de preprocessing
        """
        scaler_path = os.path.join(input_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler chargé: {scaler_path}")
            
        features_path = os.path.join(input_dir, 'selected_features.pkl')
        if os.path.exists(features_path):
            self.selected_features = joblib.load(features_path)
            logger.info(f"Features sélectionnées chargées: {features_path}")


def create_preprocessing_pipeline(data_path: str, output_dir: str = 'data/processed') -> Dict[str, Any]:
    """
    Pipeline complet de preprocessing pour données EEG
    
    Parameters:
    -----------
    data_path : str
        Chemin vers le fichier de données brutes
    output_dir : str
        Répertoire de sortie pour les données prétraitées
        
    Returns:
    --------
    dict: Informations sur le preprocessing effectué
    """
    logger.info("🧹 DÉBUT DU PIPELINE DE PREPROCESSING")
    
    # 1. Charger les données
    loader = EEGDataLoader()
    X, y = loader.load_csv_data(data_path)
    
    # 2. Valider la qualité
    quality_report = loader.validate_data_quality(X, y)
    logger.info(f"Rapport de qualité: {quality_report}")
    
    # 3. Preprocessing
    preprocessor = EEGPreprocessor()
    
    # Supprimer les features constantes
    X_clean = preprocessor.remove_constant_features(X)
    
    # Gérer les valeurs manquantes
    X_clean = preprocessor.handle_missing_values(X_clean, strategy='median')
    
    # Split train/test stratifié
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(X_clean, y)
    
    # Standardisation
    X_train_scaled, X_test_scaled = preprocessor.standardize_features(X_train, X_test)
    
    # 4. Sauvegarder les datasets
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder en format numpy (efficace)
    np.savez(os.path.join(output_dir, 'preprocessed_data.npz'),
             X_train=X_train_scaled.values, X_test=X_test_scaled.values,
             y_train=y_train.values, y_test=y_test.values)
    
    # Sauvegarder les objets de preprocessing
    preprocessor.save_preprocessing_artifacts(output_dir)
    
    # 5. Créer le rapport final
    preprocessing_report = {
        'original_shape': X.shape,
        'cleaned_shape': X_clean.shape,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'features_removed': X.shape[1] - X_clean.shape[1],
        'quality_report': quality_report,
        'preprocessing_steps': [
            'constant_feature_removal',
            'missing_value_imputation',
            'train_test_split',
            'standardization'
        ]
    }
    
    # Sauvegarder le rapport
    import json
    report_path = os.path.join(output_dir, 'preprocessing_report.json')
    with open(report_path, 'w') as f:
        # Convertir numpy types en types Python standards
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json.dump(preprocessing_report, f, indent=2, default=convert_numpy_types)
    
    logger.info(f"✅ PREPROCESSING TERMINÉ")
    logger.info(f"   • Données sauvegardées: {output_dir}/preprocessed_data.npz")
    logger.info(f"   • Rapport: {report_path}")
    logger.info(f"   • Features finales: {X_clean.shape[1]:,}")
    logger.info(f"   • Échantillons train: {len(X_train):,}")
    logger.info(f"   • Échantillons test: {len(X_test):,}")
    
    return preprocessing_report


# Fonction utilitaire pour usage rapide
def quick_preprocess(data_path: str) -> Tuple:
    """
    Preprocessing rapide pour utilisation directe
    
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test) standardisés
    """
    loader = EEGDataLoader()
    X, y = loader.load_csv_data(data_path)
    
    preprocessor = EEGPreprocessor()
    X_clean = preprocessor.remove_constant_features(X)
    X_clean = preprocessor.handle_missing_values(X_clean)
    
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(X_clean, y)
    X_train_scaled, X_test_scaled = preprocessor.standardize_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    # Test du module
    print("🧹 Test du module de preprocessing EEG")
    
    # Exemple d'utilisation
    data_path = 'C:\epilepsy-detection-project-main\Data\Raw\EEG_Scaled_data.csv'
    
    if os.path.exists(data_path):
        report = create_preprocessing_pipeline(data_path)
        print(f"✅ Pipeline terminé avec succès!")
        print(f"Rapport: {report}")
    else:
        print(f"⚠️ Fichier de test non trouvé: {data_path}")
        print("Veuillez ajuster le chemin dans la section __main__")