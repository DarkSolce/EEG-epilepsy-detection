"""
Modeling Module for EEG Epilepsy Detection
==========================================

This module provides comprehensive machine learning modeling capabilities
for EEG-based epilepsy detection with emphasis on interpretability and
clinical performance metrics.

Author: Skander Chebbi
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import joblib
import json
from collections import Counter
import logging

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, accuracy_score, precision_score, 
    recall_score, f1_score, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV

# Imbalanced learning
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class EEGModelConfig:
    """
    Configuration des modèles pour la détection d'épilepsie
    """
    
    # 🚀 FIX IMMÉDIAT POUR ACCÉLÉRER VOTRE PIPELINE
# ===============================================

# =============================================================================
# SOLUTION 1: REMPLACER get_model_configs() PAR VERSION ULTRA-RAPIDE
# =============================================================================

# Dans votre src/modeling.py, remplacez temporairement get_model_configs() par:

@staticmethod
def get_model_configs() -> Dict[str, Dict]:
    """
    Version ULTRA-RAPIDE pour gros datasets
    """
    return {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'param_grid': {  # REMPLACE param_grid par param_grid_quick
                'n_estimators': [50, 100],           # 2 valeurs au lieu de 3
                'max_depth': [10],                   # 1 valeur au lieu de 4
                'min_samples_split': [5],            # 1 valeur au lieu de 3
                'max_features': ['sqrt'],            # 1 valeur au lieu de 3
                'class_weight': ['balanced']         # 1 valeur au lieu de 3
            },
            'param_grid_quick': {
                'n_estimators': [50],
                'max_depth': [10],
                'max_features': ['sqrt']
            },
            'interpretable': True,
            'handles_imbalance': True
        },
        
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'param_grid': {  # ULTRA-RAPIDE
                'C': [1.0],                          # 1 valeur au lieu de 6
                'penalty': ['l2'],                   # 1 valeur au lieu de 3
                'class_weight': ['balanced']         # 1 valeur fixe
            },
            'param_grid_quick': {
                'C': [1.0],
                'penalty': ['l2']
            },
            'interpretable': True,
            'handles_imbalance': True
        },
        
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {  # ULTRA-RAPIDE
                'n_estimators': [50],                # 1 valeur au lieu de 3
                'learning_rate': [0.1],              # 1 valeur au lieu de 4
                'max_depth': [3]                     # 1 valeur au lieu de 3
            },
            'param_grid_quick': {
                'n_estimators': [50],
                'learning_rate': [0.1],
                'max_depth': [3]
            },
            'interpretable': True,
            'handles_imbalance': False
        }
    }

# =============================================================================
# SOLUTION 2: CRÉER config_ultra_fast.json
# =============================================================================

import json

ultra_fast_config = {
    "data": {
        "raw_data_path": "data/raw/EEG_Scaled_data.csv",
        "processed_data_dir": "data/processed_fast",
        "target_column": None
    },
    "preprocessing": {
        "test_size": 0.2,
        "random_state": 42,
        "handle_missing": "median",
        "remove_constant_features": True,
        "standardize": True
    },
    "feature_engineering": {
        "selection_method": "importance",     # AU LIEU DE "combined"
        "max_features": 300,                  # AU LIEU DE 2000
        "apply_pca": False,
        "pca_variance_threshold": 0.95
    },
    "modeling": {
        "models": ["Random Forest"],          # UN SEUL MODÈLE
        "handle_imbalance": True,
        "cv_folds": 2,                        # AU LIEU DE 5
        "scoring_metric": "f1",
        "quick_training": True                # FORCE param_grid_quick
    },
    "output": {
        "models_dir": "models_fast",
        "results_dir": "results_fast",
        "save_artifacts": True
    }
}

# Sauvegarder
with open('config_ultra_fast.json', 'w') as f:
    json.dump(ultra_fast_config, f, indent=2)

print("✅ config_ultra_fast.json créé")

# =============================================================================
# SOLUTION 3: ÉCHANTILLON STRATIFIÉ INTELLIGENT
# =============================================================================

def create_smart_sample(input_path, output_path, n_samples=5000):
    """Créer un échantillon stratifié intelligent"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print(f"📊 Création d'un échantillon stratifié de {n_samples:,} lignes...")
    
    # Lire les premières lignes pour identifier la target
    preview = pd.read_csv(input_path, nrows=1000)
    print(f"   Colonnes: {len(preview.columns)}")
    
    # Supposer que la target est la dernière colonne
    target_col = preview.columns[-1]
    print(f"   Colonne cible supposée: {target_col}")
    
    # Lire un échantillon plus large pour stratification
    print("   Lecture d'un échantillon pour stratification...")
    large_sample = pd.read_csv(input_path, nrows=20000)
    
    # Créer échantillon stratifié
    if n_samples < len(large_sample):
        X = large_sample.drop(columns=[target_col])
        y = large_sample[target_col]
        
        _, X_sample, _, y_sample = train_test_split(
            X, y, 
            test_size=n_samples/len(large_sample), 
            stratify=y, 
            random_state=42
        )
        
        # Recombiner
        sample_df = pd.concat([X_sample, y_sample], axis=1)
    else:
        sample_df = large_sample.head(n_samples)
    
    # Sauvegarder
    sample_df.to_csv(output_path, index=False)
    
    # Statistiques
    print(f"✅ Échantillon créé: {output_path}")
    print(f"   Shape: {sample_df.shape}")
    print(f"   Distribution target: {sample_df[target_col].value_counts().to_dict()}")
    print(f"   Taille fichier: {os.path.getsize(output_path)/(1024*1024):.1f} MB")
    
    return output_path

# =============================================================================
# SOLUTION 4: SCRIPT DE TEST RAPIDE
# =============================================================================

def test_pipeline_fast():
    """Test ultra-rapide du pipeline complet"""
    import os
    
    print("🚀 TEST ULTRA-RAPIDE DU PIPELINE")
    print("=" * 50)
    
    # 1. Créer échantillon si pas encore fait
    original_file = "data/raw/EEG_Scaled_data.csv"
    sample_file = "data/raw/sample_3k.csv"
    
    if not os.path.exists(sample_file):
        print("📊 Création de l'échantillon de test...")
        create_smart_sample(original_file, sample_file, n_samples=3000)
    
    # 2. Configuration ultra-rapide
    test_config = {
        "data": {"raw_data_path": sample_file, "processed_data_dir": "data/test_processed"},
        "feature_engineering": {"selection_method": "importance", "max_features": 100},
        "modeling": {"models": ["Random Forest"], "quick_training": True, "cv_folds": 2}
    }
    
    # 3. Lancer le pipeline
    try:
        print("🔄 Lancement du pipeline de test...")
        from src import create_complete_pipeline
        
        start_time = time.time()
        results = create_complete_pipeline(sample_file, test_config)
        end_time = time.time()
        
        print(f"\n✅ TEST TERMINÉ EN {end_time-start_time:.1f} SECONDES !")
        print("=" * 50)
        
        if results and 'final_performance' in results:
            best_model = results['final_performance']['best_model']
            metrics = results['final_performance']['best_metrics']
            
            print(f"🏆 Modèle: {best_model}")
            print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
            print(f"📊 Accuracy: {metrics['accuracy']:.4f}")
            print(f"📊 Precision: {metrics['precision']:.4f}")
            print(f"📊 Recall: {metrics['recall']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

# =============================================================================
# SOLUTION 5: MODIFICATION DIRECTE DU CODE
# =============================================================================

# Si vous voulez modifier directement le code au lieu du fichier config,
# dans src/modeling.py, ligne ~520 dans train_models(), ajoutez:

def train_models_ultra_fast(self, X_train, y_train):
    """Version ultra-rapide avec paramètres fixes"""
    
    # Configuration fixe ultra-rapide
    fast_models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    print("🚀 ENTRAÎNEMENT ULTRA-RAPIDE (paramètres fixes)")
    
    # Gestion déséquilibre
    if self.handle_imbalance:
        X_train_balanced, y_train_balanced = self.imbalance_handler.apply_combined_sampling(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    training_results = {}
    
    for model_name, model in fast_models.items():
        print(f"⚡ Entraînement rapide: {model_name}")
        
        # Entraînement direct sans GridSearch
        model.fit(X_train_balanced, y_train_balanced)
        
        # Validation croisée simple
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                   cv=2, scoring='f1', n_jobs=-1)
        
        self.trained_models[model_name] = model
        training_results[model_name] = {
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"✅ {model_name}: F1-Score CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return training_results

# =============================================================================
# COMMANDES À EXÉCUTER MAINTENANT
# =============================================================================

print("""
🚀 COMMANDES IMMÉDIATES À EXÉCUTER:

1️⃣ CRÉER ÉCHANTILLON DE TEST (2 minutes):
python -c "
import pandas as pd
import os
df = pd.read_csv('data/raw/EEG_Scaled_data.csv', nrows=3000)
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/sample_3k.csv', index=False)
print(f'✅ Échantillon créé: {df.shape}')
print(f'Distribution: {df.iloc[:, -1].value_counts().to_dict()}')
"

2️⃣ TEST ULTRA-RAPIDE (5 minutes max):
python main.py --data data/raw/sample_3k.csv --quick

3️⃣ SI ÇA MARCHE, DATASET COMPLET AVEC CONFIG RAPIDE:
python main.py --config config_ultra_fast.json

4️⃣ TEMPS ESTIMÉS:
   - Échantillon 3k: 3-5 minutes
   - Dataset complet optimisé: 30-45 minutes
   - Dataset complet original: 2-3 heures

⚠️  IMPORTANT: 
   Le mode --quick devrait automatiquement utiliser param_grid_quick
   Si ça ne marche pas, modifiez temporairement param_grid dans get_model_configs()
""")

import time
import os

# Créer immédiatement la config ultra-rapide
if __name__ == "__main__":
    # Créer le fichier de config
    with open('config_ultra_fast.json', 'w') as f:
        json.dump(ultra_fast_config, f, indent=2)
    print("✅ config_ultra_fast.json créé et prêt à utiliser")


class ImbalanceHandler:
    """
    Gestionnaire du déséquilibre des classes pour l'épilepsie
    """
    
    def __init__(self):
        self.samplers = {}
        self.sampling_strategy = None
        
    def apply_smote(self, X: np.ndarray, y: np.ndarray, 
                   sampling_strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Appliquer SMOTE pour rééquilibrer les classes
        """
        if not IMBLEARN_AVAILABLE:
            logger.warning("imblearn non disponible, pas de rééquilibrage")
            return X, y
            
        logger.info(f"SMOTE avec stratégie: {sampling_strategy}")
        original_distribution = Counter(y)
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        new_distribution = Counter(y_resampled)
        self.samplers['smote'] = smote
        
        logger.info(f"Distribution avant: {original_distribution}")
        logger.info(f"Distribution après: {new_distribution}")
        
        return X_resampled, y_resampled
    
    def apply_combined_sampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combinaison SMOTE + sous-échantillonnage
        """
        if not IMBLEARN_AVAILABLE:
            return X, y
            
        logger.info("Échantillonnage combiné SMOTE + Under-sampling")
        
        # D'abord SMOTE pour augmenter la classe minoritaire
        smote = SMOTE(sampling_strategy=0.5, random_state=42)  # 1:2 ratio
        X_smote, y_smote = smote.fit_resample(X, y)
        
        # Puis sous-échantillonnage pour réduire la classe majoritaire
        undersampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # 1:1.25 ratio
        X_final, y_final = undersampler.fit_resample(X_smote, y_smote)
        
        self.samplers['combined'] = (smote, undersampler)
        
        logger.info(f"Distribution finale: {Counter(y_final)}")
        return X_final, y_final
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculer les poids de classes pour modèles supportant class_weight
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        logger.info(f"Poids de classes calculés: {class_weights}")
        return class_weights


class ModelEvaluator:
    """
    Évaluateur de modèles avec métriques cliniques
    """
    
    def __init__(self):
        self.evaluation_results = {}
        
    def compute_clinical_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculer les métriques cliniques importantes pour l'épilepsie
        """
        # Métriques de base
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._compute_specificity(y_true, y_pred)
        }
        
        # Métriques avec probabilités
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'average_precision': average_precision_score(y_true, y_pred_proba)
            })
        
        # Métriques cliniques spécifiques
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Valeur prédictive positive (précision clinique)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Valeur prédictive négative
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Likelihood ratios
        sensitivity = metrics['recall']
        specificity = metrics['specificity']
        
        lr_positive = sensitivity / (1 - specificity) if specificity < 1.0 else float('inf')
        lr_negative = (1 - sensitivity) / specificity if specificity > 0.0 else float('inf')
        
        metrics.update({
            'ppv': ppv,
            'npv': npv,
            'lr_positive': lr_positive,
            'lr_negative': lr_negative,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        })
        
        return metrics
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculer la spécificité (Taux de vrais négatifs)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def evaluate_model_comprehensive(self, model, X_test: np.ndarray, y_test: np.ndarray,
                                   model_name: str = "Unknown") -> Dict[str, Any]:
        """
        Évaluation complète d'un modèle avec métriques cliniques
        """
        logger.info(f"Évaluation complète du modèle: {model_name}")
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Probabilités si disponibles
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None
            logger.warning(f"Probabilités non disponibles pour {model_name}")
        
        # Métriques cliniques
        clinical_metrics = self.compute_clinical_metrics(y_test, y_pred, y_pred_proba)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Courbes si probabilités disponibles
        curves_data = {}
        if y_pred_proba is not None:
            # Courbe ROC
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
            curves_data['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }
            
            # Courbe Précision-Rappel
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
            curves_data['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        
        # Rapport complet
        evaluation_report = {
            'model_name': model_name,
            'test_samples': len(y_test),
            'class_distribution': dict(Counter(y_test)),
            'clinical_metrics': clinical_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'curves_data': curves_data
        }
        
        self.evaluation_results[model_name] = evaluation_report
        
        # Log des métriques principales
        logger.info(f"Résultats pour {model_name}:")
        logger.info(f"  • Accuracy: {clinical_metrics['accuracy']:.4f}")
        logger.info(f"  • Precision: {clinical_metrics['precision']:.4f}")
        logger.info(f"  • Recall (Sensitivity): {clinical_metrics['recall']:.4f}")
        logger.info(f"  • Specificity: {clinical_metrics['specificity']:.4f}")
        logger.info(f"  • F1-Score: {clinical_metrics['f1_score']:.4f}")
        
        return evaluation_report
    
    def compare_models(self) -> pd.DataFrame:
        """
        Comparer tous les modèles évalués
        """
        if not self.evaluation_results:
            logger.warning("Aucun modèle évalué")
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            metrics = results['clinical_metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Specificity': metrics['specificity'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', 'N/A'),
                'PPV': metrics['ppv'],
                'NPV': metrics['npv']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('F1-Score', ascending=False)


class ThresholdOptimizer:
    """
    Optimiseur de seuil de classification pour maximiser les métriques cliniques
    """
    
    def __init__(self):
        self.optimal_thresholds = {}
        
    def optimize_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                          metric: str = 'f1', thresholds: np.ndarray = None) -> Dict[str, float]:
        """
        Optimiser le seuil de classification
        
        Parameters:
        -----------
        y_true : array
            Vraies étiquettes
        y_pred_proba : array
            Probabilités prédites
        metric : str
            Métrique à optimiser ('f1', 'precision', 'recall', 'accuracy')
        thresholds : array
            Seuils à tester (par défaut: np.arange(0.1, 0.9, 0.01))
            
        Returns:
        --------
        dict: Informations sur le seuil optimal
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.01)
        
        logger.info(f"Optimisation du seuil pour métrique: {metric}")
        
        best_threshold = 0.5
        best_score = 0.0
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Éviter la division par zéro
            if len(np.unique(y_pred_thresh)) == 1:
                continue
            
            # Calculer la métrique
            if metric == 'f1':
                score = f1_score(y_true, y_pred_thresh, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred_thresh, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred_thresh, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred_thresh)
            else:
                raise ValueError(f"Métrique non supportée: {metric}")
            
            threshold_results.append({
                'threshold': threshold,
                'score': score,
                'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
                'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
                'f1': f1_score(y_true, y_pred_thresh, zero_division=0)
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Calculer les métriques pour le seuil optimal
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        evaluator = ModelEvaluator()
        optimal_metrics = evaluator.compute_clinical_metrics(y_true, y_pred_optimal, y_pred_proba)
        
        optimization_result = {
            'optimal_threshold': best_threshold,
            'optimal_score': best_score,
            'optimized_metric': metric,
            'optimal_metrics': optimal_metrics,
            'threshold_analysis': threshold_results
        }
        
        logger.info(f"Seuil optimal: {best_threshold:.3f} (Score: {best_score:.4f})")
        
        return optimization_result


class EEGModelTrainer:
    """
    Entraîneur de modèles principal pour la détection d'épilepsie
    """
    
    def __init__(self, handle_imbalance: bool = True):
        self.models = {}
        self.trained_models = {}
        self.optimization_results = {}
        self.handle_imbalance = handle_imbalance
        self.imbalance_handler = ImbalanceHandler()
        self.evaluator = ModelEvaluator()
        self.threshold_optimizer = ThresholdOptimizer()
        
    def setup_models(self, model_names: List[str] = None, quick_search: bool = False) -> None:
        """
        Configurer les modèles à entraîner
        """
        if model_names is None:
            model_names = ['Random Forest', 'Logistic Regression', 'Gradient Boosting']
        
        model_configs = EEGModelConfig.get_model_configs()
        
        for name in model_names:
            if name in model_configs:
                config = model_configs[name].copy()
                # Utiliser param_grid_quick si demandé
                if quick_search and 'param_grid_quick' in config:
                    config['param_grid'] = config['param_grid_quick']
                
                self.models[name] = config
                logger.info(f"Modèle configuré: {name}")
        
        logger.info(f"Total modèles configurés: {len(self.models)}")
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    cv_folds: int = 5, scoring: str = 'f1',
                    n_jobs: int = -1, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîner tous les modèles configurés avec optimisation d'hyperparamètres
        """
        logger.info("🚀 DÉBUT DE L'ENTRAÎNEMENT DES MODÈLES")
        logger.info(f"Dataset: {X_train.shape[0]} échantillons, {X_train.shape[1]} features")
        logger.info(f"Distribution: {Counter(y_train)}")
        
        # Gérer le déséquilibre si demandé
        if self.handle_imbalance:
            X_train_balanced, y_train_balanced = self.imbalance_handler.apply_combined_sampling(X_train, y_train)
            logger.info(f"Après rééquilibrage: {Counter(y_train_balanced)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Configuration de la validation croisée
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        training_results = {}
        
        for model_name, model_config in self.models.items():
            logger.info(f"\n🔄 Entraînement: {model_name}")
            
            try:
                # Optimisation des hyperparamètres
                grid_search = GridSearchCV(
                    estimator=model_config['model'],
                    param_grid=model_config['param_grid'],
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=1 if verbose else 0
                )
                
                # Entraînement
                grid_search.fit(X_train_balanced, y_train_balanced)
                
                # Sauvegarder le meilleur modèle
                best_model = grid_search.best_estimator_
                self.trained_models[model_name] = best_model
                
                # Résultats d'optimisation
                optimization_result = {
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'cv_results': {
                        'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                        'std_test_scores': grid_search.cv_results_['std_test_score'].tolist()
                    }
                }
                
                self.optimization_results[model_name] = optimization_result
                training_results[model_name] = optimization_result
                
                logger.info(f"✅ {model_name} - Meilleur score CV: {grid_search.best_score_:.4f}")
                logger.info(f"   Meilleurs paramètres: {grid_search.best_params_}")
                
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'entraînement de {model_name}: {str(e)}")
                continue
        
        logger.info(f"✅ ENTRAÎNEMENT TERMINÉ - {len(self.trained_models)} modèles entraînés")
        
        return training_results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Évaluer tous les modèles entraînés
        """
        logger.info("📊 ÉVALUATION DES MODÈLES")
        
        evaluation_results = {}
        
        for model_name, model in self.trained_models.items():
            logger.info(f"Évaluation: {model_name}")
            
            # Évaluation complète
            eval_result = self.evaluator.evaluate_model_comprehensive(
                model, X_test, y_test, model_name
            )
            evaluation_results[model_name] = eval_result
            
            # Optimisation du seuil si probabilités disponibles
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                threshold_result = self.threshold_optimizer.optimize_threshold(
                    y_test, y_pred_proba, metric='f1'
                )
                eval_result['threshold_optimization'] = threshold_result
                logger.info(f"   Seuil optimal: {threshold_result['optimal_threshold']:.3f}")
            except:
                logger.info(f"   Optimisation de seuil non applicable pour {model_name}")
        
        return evaluation_results
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Récupérer le meilleur modèle selon une métrique
        """
        if not self.evaluator.evaluation_results:
            raise ValueError("Aucun modèle évalué")
        
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.evaluator.evaluation_results.items():
            score = results['clinical_metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError(f"Aucun modèle trouvé avec la métrique: {metric}")
        
        return best_model_name, self.trained_models[best_model_name]
    
    def generate_interpretability_report(self, model_name: str, 
                                       feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Générer un rapport d'interprétabilité pour un modèle
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle non trouvé: {model_name}")
        
        model = self.trained_models[model_name]
        interpretability_report = {
            'model_name': model_name,
            'model_type': type(model).__name__
        }
        
        # Feature importance pour RF et GB
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is not None:
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            else:
                feature_importance_df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importances))],
                    'importance': importances
                }).sort_values('importance', ascending=False)
            
            interpretability_report['feature_importance'] = {
                'top_20': feature_importance_df.head(20).to_dict('records'),
                'importance_stats': {
                    'mean': float(np.mean(importances)),
                    'std': float(np.std(importances)),
                    'max': float(np.max(importances)),
                    'min': float(np.min(importances))
                }
            }
        
        # Coefficients pour la régression logistique
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            
            if feature_names is not None:
                coef_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefficients,
                    'abs_coefficient': np.abs(coefficients)
                }).sort_values('abs_coefficient', ascending=False)
            else:
                coef_df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(coefficients))],
                    'coefficient': coefficients,
                    'abs_coefficient': np.abs(coefficients)
                }).sort_values('abs_coefficient', ascending=False)
            
            interpretability_report['coefficients'] = {
                'top_positive': coef_df[coef_df['coefficient'] > 0].head(10).to_dict('records'),
                'top_negative': coef_df[coef_df['coefficient'] < 0].head(10).to_dict('records'),
                'coefficient_stats': {
                    'mean': float(np.mean(coefficients)),
                    'std': float(np.std(coefficients)),
                    'max': float(np.max(coefficients)),
                    'min': float(np.min(coefficients))
                }
            }
        
        return interpretability_report
    
    def save_models_and_results(self, output_dir: str = 'models') -> None:
        """
        Sauvegarder tous les modèles et résultats
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder les modèles
        for model_name, model in self.trained_models.items():
            model_filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            model_path = os.path.join(output_dir, model_filename)
            joblib.dump(model, model_path)
            logger.info(f"Modèle sauvegardé: {model_path}")
        
        # Sauvegarder les résultats
        results_summary = {
            'training_config': {
                'handle_imbalance': self.handle_imbalance,
                'models_trained': list(self.trained_models.keys())
            },
            'optimization_results': self.optimization_results,
            'evaluation_results': {
                name: {k: v for k, v in results.items() 
                      if k not in ['curves_data']}  # Exclure les courbes pour la taille
                for name, results in self.evaluator.evaluation_results.items()
            }
        }
        
        results_path = os.path.join(output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"Résultats sauvegardés: {results_path}")
        
        # Sauvegarder le comparatif des modèles
        comparison_df = self.evaluator.compare_models()
        if not comparison_df.empty:
            comparison_path = os.path.join(output_dir, 'model_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"Comparatif sauvegardé: {comparison_path}")


def create_complete_modeling_pipeline(X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray,
                                    model_names: List[str] = None,
                                    quick_training: bool = False,
                                    output_dir: str = 'models') -> Dict[str, Any]:
    """
    Pipeline complet de modélisation pour détection d'épilepsie
    
    Parameters:
    -----------
    X_train, X_test : np.ndarray
        Datasets d'entraînement et de test
    y_train, y_test : np.ndarray
        Labels
    model_names : list
        Modèles à entraîner
    quick_training : bool
        Utiliser une recherche d'hyperparamètres rapide
    output_dir : str
        Répertoire de sauvegarde
        
    Returns:
    --------
    dict: Résultats complets du pipeline
    """
    logger.info("🚀 PIPELINE COMPLET DE MODÉLISATION EEG")
    
    # Initialiser le trainer
    trainer = EEGModelTrainer(handle_imbalance=True)
    
    # Configurer les modèles
    if model_names is None:
        model_names = ['Random Forest', 'Logistic Regression', 'Gradient Boosting']
    
    trainer.setup_models(model_names, quick_search=quick_training)
    
    # Entraînement
    training_results = trainer.train_models(X_train, y_train, cv_folds=5)
    
    # Évaluation
    evaluation_results = trainer.evaluate_models(X_test, y_test)
    
    # Identifier le meilleur modèle
    best_model_name, best_model = trainer.get_best_model(metric='f1_score')
    
    # Rapport d'interprétabilité
    interpretability_report = trainer.generate_interpretability_report(best_model_name)
    
    # Sauvegarder
    trainer.save_models_and_results(output_dir)
    
    # Résumé final
    pipeline_results = {
        'best_model': best_model_name,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'interpretability_report': interpretability_report,
        'model_comparison': trainer.evaluator.compare_models().to_dict('records'),
        'pipeline_config': {
            'models_trained': model_names,
            'quick_training': quick_training,
            'handle_imbalance': True,
            'train_samples': len(y_train),
            'test_samples': len(y_test)
        }
    }
    
    logger.info(f"✅ PIPELINE TERMINÉ")
    logger.info(f"   • Meilleur modèle: {best_model_name}")
    logger.info(f"   • Modèles sauvegardés dans: {output_dir}")
    
    return pipeline_results


if __name__ == '__main__':
    # Test du module
    print("🤖 Test du module de modélisation EEG")
    
    # Générer des données de test
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    X_train_test = np.random.randn(n_samples, n_features)
    y_train_test = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    X_test_test = np.random.randn(200, n_features)
    y_test_test = np.random.choice([0, 1], size=200, p=[0.8, 0.2])
    
    print(f"Dataset de test: Train={X_train_test.shape}, Test={X_test_test.shape}")
    
    # Test du pipeline rapide
    try:
        results = create_complete_modeling_pipeline(
            X_train_test, y_train_test, X_test_test, y_test_test,
            model_names=['Random Forest', 'Logistic Regression'],
            quick_training=True,
            output_dir='test_models'
        )
        print(f"✅ Pipeline testé - Meilleur modèle: {results['best_model']}")
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")