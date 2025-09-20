"""
Feature Engineering Module for EEG Epilepsy Detection
=====================================================

This module provides advanced feature selection and engineering techniques
specifically designed for EEG data analysis and epilepsy detection.

Author: Skander Chebbi
Date: 2025
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_classif, chi2, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Dict, Any, Optional
import joblib
import logging

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Classe pour la sélection de features EEG avec plusieurs méthodes
    """
    
    def __init__(self):
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selection_method_ = None
        self.selector_objects_ = {}
        
    def univariate_selection(self, X: pd.DataFrame, y: pd.Series, 
                           score_func=f_classif, k: int = 1000) -> pd.DataFrame:
        """
        Sélection univariée basée sur les scores statistiques
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features EEG
        y : pd.Series
            Labels
        score_func : callable
            Fonction de score (f_classif, chi2, mutual_info_classif)
        k : int
            Nombre de features à sélectionner
            
        Returns:
        --------
        pd.DataFrame: Features sélectionnées
        """
        logger.info(f"Sélection univariée: {score_func.__name__}, k={k}")
        
        # Ajuster k au nombre de features disponibles
        k = min(k, X.shape[1])
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Récupérer les noms des features sélectionnées
        selected_features = X.columns[selector.get_support()]
        
        # Sauvegarder les informations
        self.selector_objects_['univariate'] = selector
        self.feature_scores_ = pd.Series(selector.scores_, index=X.columns)
        
        logger.info(f"Features sélectionnées: {len(selected_features):,}")
        logger.info(f"Score moyen: {selector.scores_.mean():.4f}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    estimator=None, n_features: int = 500,
                                    cv: int = 5) -> pd.DataFrame:
        """
        Élimination récursive de features (RFE/RFECV)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features EEG
        y : pd.Series
            Labels
        estimator : sklearn estimator
            Modèle pour évaluer l'importance des features
        n_features : int
            Nombre de features à sélectionner
        cv : int
            Nombre de folds pour la validation croisée (si RFECV)
            
        Returns:
        --------
        pd.DataFrame: Features sélectionnées
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
        logger.info(f"RFE avec {type(estimator).__name__}, n_features={n_features}")
        
        # Ajuster le nombre de features
        n_features = min(n_features, X.shape[1])
        
        if cv > 1:
            # Utiliser RFECV pour optimiser automatiquement le nombre de features
            selector = RFECV(
                estimator=estimator,
                step=0.1,  # Supprimer 10% des features à chaque itération
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring='f1',
                n_jobs=-1
            )
            logger.info("Utilisation de RFECV avec validation croisée")
        else:
            # RFE simple
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
            logger.info("Utilisation de RFE simple")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.support_]
        
        # Sauvegarder
        self.selector_objects_['rfe'] = selector
        
        if hasattr(selector, 'n_features_'):
            optimal_features = selector.n_features_
            logger.info(f"Nombre optimal de features (RFECV): {optimal_features}")
        
        logger.info(f"Features finalement sélectionnées: {len(selected_features):,}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def importance_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 estimator=None, threshold: float = None,
                                 top_k: int = None) -> pd.DataFrame:
        """
        Sélection basée sur l'importance des features (Random Forest, etc.)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features EEG
        y : pd.Series
            Labels
        estimator : sklearn estimator
            Modèle avec feature_importances_
        threshold : float
            Seuil d'importance minimum
        top_k : int
            Prendre les k features les plus importantes
            
        Returns:
        --------
        pd.DataFrame: Features sélectionnées
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            
        logger.info(f"Sélection par importance avec {type(estimator).__name__}")
        
        # Entraîner le modèle
        estimator.fit(X, y)
        
        # Récupérer les importances
        importances = estimator.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Sélectionner les features selon le critère
        if top_k is not None:
            selected_features = feature_importance_df.head(top_k)['feature'].values
            logger.info(f"Top {top_k} features sélectionnées")
        elif threshold is not None:
            selected_features = feature_importance_df[
                feature_importance_df['importance'] >= threshold
            ]['feature'].values
            logger.info(f"Features avec importance >= {threshold}: {len(selected_features)}")
        else:
            # Par défaut, prendre les features avec importance > importance moyenne
            mean_importance = importances.mean()
            selected_features = feature_importance_df[
                feature_importance_df['importance'] > mean_importance
            ]['feature'].values
            logger.info(f"Features avec importance > moyenne ({mean_importance:.6f}): {len(selected_features)}")
        
        # Sauvegarder
        self.selector_objects_['importance'] = estimator
        self.feature_scores_ = pd.Series(importances, index=X.columns)
        
        return X[selected_features]
    
    def combined_selection(self, X: pd.DataFrame, y: pd.Series, 
                          methods: List[str] = None) -> pd.DataFrame:
        """
        Combiner plusieurs méthodes de sélection pour robustesse
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features EEG
        y : pd.Series
            Labels
        methods : list
            Liste des méthodes à combiner
            
        Returns:
        --------
        pd.DataFrame: Features sélectionnées par consensus
        """
        if methods is None:
            methods = ['univariate', 'importance', 'rfe']
            
        logger.info(f"Sélection combinée avec méthodes: {methods}")
        
        selected_sets = {}
        
        # Appliquer chaque méthode
        if 'univariate' in methods:
            X_univ = self.univariate_selection(X, y, k=min(2000, X.shape[1]))
            selected_sets['univariate'] = set(X_univ.columns)
            
        if 'importance' in methods:
            X_imp = self.importance_based_selection(X, y, top_k=min(1500, X.shape[1]))
            selected_sets['importance'] = set(X_imp.columns)
            
        if 'rfe' in methods:
            X_rfe = self.recursive_feature_elimination(X, y, n_features=min(1000, X.shape[1]), cv=3)
            selected_sets['rfe'] = set(X_rfe.columns)
        
        # Trouver l'intersection (features sélectionnées par plusieurs méthodes)
        all_features = set()
        for features in selected_sets.values():
            all_features.update(features)
        
        # Compter combien de méthodes ont sélectionné chaque feature
        feature_votes = {}
        for feature in all_features:
            votes = sum(1 for features in selected_sets.values() if feature in features)
            feature_votes[feature] = votes
        
        # Sélectionner les features avec au moins 2 votes (ou 1 si pas assez)
        min_votes = 2 if len(methods) >= 2 else 1
        consensus_features = [f for f, votes in feature_votes.items() if votes >= min_votes]
        
        # Si pas assez de features, prendre celles avec le plus de votes
        if len(consensus_features) < 100:
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            consensus_features = [f for f, _ in sorted_features[:min(1000, len(sorted_features))]]
        
        self.selected_features_ = consensus_features
        self.selection_method_ = f"combined_{'+'.join(methods)}"
        
        logger.info(f"Features consensus: {len(consensus_features):,}")
        logger.info(f"Répartition des votes: {dict(pd.Series(list(feature_votes.values())).value_counts())}")
        
        return X[consensus_features]
    
    def get_feature_ranking(self, method: str = 'importance') -> pd.DataFrame:
        """
        Obtenir le classement des features selon une méthode
        """
        if method not in self.selector_objects_:
            raise ValueError(f"Méthode {method} non disponible. Exécutez d'abord la sélection.")
            
        if method == 'importance':
            estimator = self.selector_objects_[method]
            importance_df = pd.DataFrame({
                'feature': estimator.feature_names_in_ if hasattr(estimator, 'feature_names_in_') else range(len(estimator.feature_importances_)),
                'importance': estimator.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        
        elif method == 'univariate':
            selector = self.selector_objects_[method]
            score_df = pd.DataFrame({
                'feature': selector.feature_names_in_ if hasattr(selector, 'feature_names_in_') else range(len(selector.scores_)),
                'score': selector.scores_
            }).sort_values('score', ascending=False)
            return score_df
        
        return pd.DataFrame()


class DimensionalityReducer:
    """
    Réduction de dimensionnalité pour les données EEG
    """
    
    def __init__(self):
        self.reducer = None
        self.explained_variance_ratio_ = None
        
    def apply_pca(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                  variance_threshold: float = 0.95) -> Tuple:
        """
        Appliquer PCA pour réduction dimensionnelle
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Dataset d'entraînement
        X_test : pd.DataFrame, optional
            Dataset de test
        variance_threshold : float
            Pourcentage de variance à conserver
            
        Returns:
        --------
        tuple: Datasets transformés
        """
        logger.info(f"PCA avec seuil de variance: {variance_threshold}")
        
        # Déterminer le nombre de composantes
        pca_temp = PCA()
        pca_temp.fit(X_train)
        
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        logger.info(f"Composantes nécessaires pour {variance_threshold:.1%} variance: {n_components}")
        
        # Appliquer PCA avec le nombre optimal de composantes
        self.reducer = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.reducer.fit_transform(X_train)
        
        # Créer les noms des composantes
        component_names = [f'PC_{i+1}' for i in range(n_components)]
        X_train_pca = pd.DataFrame(X_train_pca, columns=component_names, index=X_train.index)
        
        self.explained_variance_ratio_ = self.reducer.explained_variance_ratio_
        
        if X_test is not None:
            X_test_pca = self.reducer.transform(X_test)
            X_test_pca = pd.DataFrame(X_test_pca, columns=component_names, index=X_test.index)
            
            logger.info(f"PCA appliquée: {X_train.shape} → {X_train_pca.shape}")
            return X_train_pca, X_test_pca
        else:
            return X_train_pca
    
    def get_component_analysis(self, feature_names: List[str] = None, top_k: int = 10) -> Dict:
        """
        Analyser les composantes principales
        """
        if self.reducer is None:
            raise ValueError("PCA non encore appliquée")
            
        analysis = {
            'explained_variance_ratio': self.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(self.explained_variance_ratio_).tolist(),
            'n_components': len(self.explained_variance_ratio_)
        }
        
        if feature_names is not None:
            # Analyser les contributions des features originales
            components_df = pd.DataFrame(
                self.reducer.components_.T,
                index=feature_names,
                columns=[f'PC_{i+1}' for i in range(len(self.explained_variance_ratio_))]
            )
            
            # Top features par composante
            top_contributors = {}
            for i, pc in enumerate(components_df.columns[:min(5, len(components_df.columns))]):
                top_pos = components_df[pc].nlargest(top_k)
                top_neg = components_df[pc].nsmallest(top_k)
                top_contributors[pc] = {
                    'positive': dict(top_pos),
                    'negative': dict(top_neg)
                }
            
            analysis['top_contributors'] = top_contributors
        
        return analysis


class EEGFeatureEngineer:
    """
    Classe principale pour l'ingénierie des features EEG
    """
    
    def __init__(self):
        self.feature_selector = FeatureSelector()
        self.dim_reducer = DimensionalityReducer()
        self.selected_features = None
        self.engineering_report = {}
        
    def engineer_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame = None,
                         selection_method: str = 'combined',
                         apply_pca: bool = False,
                         pca_variance_threshold: float = 0.95) -> Tuple:
        """
        Pipeline complet d'ingénierie des features
        
        Parameters:
        -----------
        X_train, X_test : pd.DataFrame
            Datasets d'entraînement et de test
        y_train : pd.Series
            Labels d'entraînement
        selection_method : str
            Méthode de sélection ('univariate', 'importance', 'rfe', 'combined')
        apply_pca : bool
            Appliquer la réduction PCA après sélection
        pca_variance_threshold : float
            Seuil de variance pour PCA
            
        Returns:
        --------
        tuple: Datasets avec features engineering
        """
        logger.info(f"🔧 DÉBUT DE L'INGÉNIERIE DES FEATURES")
        logger.info(f"   Méthode: {selection_method}")
        logger.info(f"   PCA: {apply_pca}")
        logger.info(f"   Shape initiale train: {X_train.shape}")
        
        original_shape = X_train.shape
        
        # 1. Sélection de features
        if selection_method == 'combined':
            X_train_selected = self.feature_selector.combined_selection(X_train, y_train)
        elif selection_method == 'univariate':
            X_train_selected = self.feature_selector.univariate_selection(
                X_train, y_train, k=min(2000, X_train.shape[1])
            )
        elif selection_method == 'importance':
            X_train_selected = self.feature_selector.importance_based_selection(
                X_train, y_train, top_k=min(1500, X_train.shape[1])
            )
        elif selection_method == 'rfe':
            X_train_selected = self.feature_selector.recursive_feature_elimination(
                X_train, y_train, n_features=min(1000, X_train.shape[1])
            )
        else:
            raise ValueError(f"Méthode non supportée: {selection_method}")
        
        # Appliquer la même sélection au test
        selected_features = X_train_selected.columns.tolist()
        self.selected_features = selected_features
        
        if X_test is not None:
            X_test_selected = X_test[selected_features]
        else:
            X_test_selected = None
        
        selection_shape = X_train_selected.shape
        logger.info(f"   Shape après sélection: {selection_shape}")
        
        # 2. Réduction dimensionnelle (optionnelle)
        if apply_pca:
            if X_test_selected is not None:
                X_train_final, X_test_final = self.dim_reducer.apply_pca(
                    X_train_selected, X_test_selected, 
                    variance_threshold=pca_variance_threshold
                )
            else:
                X_train_final = self.dim_reducer.apply_pca(
                    X_train_selected, variance_threshold=pca_variance_threshold
                )
                X_test_final = None
            
            final_shape = X_train_final.shape
            logger.info(f"   Shape après PCA: {final_shape}")
            
            # Analyse des composantes
            pca_analysis = self.dim_reducer.get_component_analysis(selected_features)
            self.engineering_report['pca_analysis'] = pca_analysis
        else:
            X_train_final = X_train_selected
            X_test_final = X_test_selected
            final_shape = X_train_final.shape
        
        # 3. Créer le rapport d'ingénierie
        self.engineering_report.update({
            'original_features': original_shape[1],
            'selected_features': selection_shape[1],
            'final_features': final_shape[1],
            'selection_method': selection_method,
            'pca_applied': apply_pca,
            'reduction_ratio': 1 - (final_shape[1] / original_shape[1]),
            'selected_feature_names': selected_features[:100]  # Top 100 pour le rapport
        })
        
        logger.info(f"✅ INGÉNIERIE TERMINÉE")
        logger.info(f"   • Réduction: {original_shape[1]:,} → {final_shape[1]:,} features")
        logger.info(f"   • Ratio de réduction: {self.engineering_report['reduction_ratio']:.1%}")
        
        return X_train_final, X_test_final
    
    def get_feature_importance_report(self) -> Dict:
        """
        Générer un rapport d'importance des features
        """
        report = {}
        
        if 'importance' in self.feature_selector.selector_objects_:
            importance_ranking = self.feature_selector.get_feature_ranking('importance')
            report['importance_ranking'] = importance_ranking.head(50).to_dict('records')
        
        if 'univariate' in self.feature_selector.selector_objects_:
            univariate_ranking = self.feature_selector.get_feature_ranking('univariate')
            report['univariate_ranking'] = univariate_ranking.head(50).to_dict('records')
        
        if hasattr(self.feature_selector, 'feature_scores_') and self.feature_selector.feature_scores_ is not None:
            top_scores = self.feature_selector.feature_scores_.nlargest(20)
            report['top_feature_scores'] = dict(top_scores)
        
        return report
    
    def save_engineering_artifacts(self, output_dir: str = 'data/processed') -> None:
        """
        Sauvegarder les objets d'ingénierie des features
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder le sélecteur de features
        if self.feature_selector.selected_features_ is not None:
            joblib.dump(self.feature_selector, 
                       os.path.join(output_dir, 'feature_selector.pkl'))
            logger.info("Feature selector sauvegardé")
        
        # Sauvegarder le réducteur de dimensionnalité
        if self.dim_reducer.reducer is not None:
            joblib.dump(self.dim_reducer, 
                       os.path.join(output_dir, 'dimension_reducer.pkl'))
            logger.info("Dimension reducer sauvegardé")
        
        # Sauvegarder les features sélectionnées
        if self.selected_features is not None:
            with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
                for feature in self.selected_features:
                    f.write(f"{feature}\n")
            logger.info(f"Liste des {len(self.selected_features)} features sélectionnées sauvegardée")
        
        # Sauvegarder le rapport d'ingénierie
        if self.engineering_report:
            import json
            with open(os.path.join(output_dir, 'feature_engineering_report.json'), 'w') as f:
                json.dump(self.engineering_report, f, indent=2, default=str)
            logger.info("Rapport d'ingénierie sauvegardé")
    
    def load_engineering_artifacts(self, input_dir: str = 'data/processed') -> None:
        """
        Charger les objets d'ingénierie
        """
        import os
        
        # Charger le sélecteur
        selector_path = os.path.join(input_dir, 'feature_selector.pkl')
        if os.path.exists(selector_path):
            self.feature_selector = joblib.load(selector_path)
            logger.info("Feature selector chargé")
        
        # Charger le réducteur
        reducer_path = os.path.join(input_dir, 'dimension_reducer.pkl')
        if os.path.exists(reducer_path):
            self.dim_reducer = joblib.load(reducer_path)
            logger.info("Dimension reducer chargé")
        
        # Charger la liste des features
        features_path = os.path.join(input_dir, 'selected_features.txt')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.selected_features = [line.strip() for line in f.readlines()]
            logger.info(f"{len(self.selected_features)} features chargées")


def create_feature_engineering_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                                      X_test: pd.DataFrame = None,
                                      method: str = 'combined',
                                      output_dir: str = 'data/processed') -> Tuple:
    """
    Pipeline complet d'ingénierie des features EEG
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Datasets preprocessés
    y_train : pd.Series
        Labels
    method : str
        Méthode de sélection
    output_dir : str
        Répertoire de sauvegarde
        
    Returns:
    --------
    tuple: (X_train_engineered, X_test_engineered, feature_report)
    """
    logger.info("🔧 PIPELINE D'INGÉNIERIE DES FEATURES EEG")
    
    # Créer l'ingénieur de features
    engineer = EEGFeatureEngineer()
    
    # Appliquer l'ingénierie
    X_train_eng, X_test_eng = engineer.engineer_features(
        X_train, y_train, X_test,
        selection_method=method,
        apply_pca=False,  # PCA séparé si nécessaire
        pca_variance_threshold=0.95
    )
    
    # Générer les rapports
    importance_report = engineer.get_feature_importance_report()
    engineering_report = engineer.engineering_report
    engineering_report['importance_analysis'] = importance_report
    
    # Sauvegarder
    engineer.save_engineering_artifacts(output_dir)
    
    logger.info("✅ PIPELINE D'INGÉNIERIE TERMINÉ")
    
    return X_train_eng, X_test_eng, engineering_report


# Fonctions utilitaires
def quick_feature_selection(X: pd.DataFrame, y: pd.Series, 
                          method: str = 'importance', n_features: int = 1000) -> pd.DataFrame:
    """
    Sélection rapide de features pour prototypage
    """
    selector = FeatureSelector()
    
    if method == 'importance':
        return selector.importance_based_selection(X, y, top_k=n_features)
    elif method == 'univariate':
        return selector.univariate_selection(X, y, k=n_features)
    elif method == 'rfe':
        return selector.recursive_feature_elimination(X, y, n_features=n_features, cv=1)
    else:
        raise ValueError(f"Méthode non supportée: {method}")


def analyze_feature_correlation(X: pd.DataFrame, threshold: float = 0.95) -> Dict:
    """
    Analyser la corrélation entre features pour détecter la redondance
    """
    logger.info(f"Analyse des corrélations (seuil: {threshold})")
    
    # Calculer la matrice de corrélation sur un échantillon si trop large
    if X.shape[1] > 5000:
        sample_features = X.sample(n=5000, axis=1, random_state=42)
        logger.info(f"Échantillonnage: {sample_features.shape[1]} features")
        corr_matrix = sample_features.corr()
    else:
        corr_matrix = X.corr()
    
    # Trouver les paires hautement corrélées
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val >= threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    # Statistiques des corrélations
    corr_values = corr_matrix.values
    upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
    
    correlation_stats = {
        'mean_correlation': float(np.mean(abs(upper_triangle))),
        'max_correlation': float(np.max(abs(upper_triangle))),
        'high_correlation_pairs': len(high_corr_pairs),
        'high_correlation_threshold': threshold,
        'correlation_distribution': {
            'very_low': int(np.sum(abs(upper_triangle) < 0.1)),
            'low': int(np.sum((abs(upper_triangle) >= 0.1) & (abs(upper_triangle) < 0.3))),
            'moderate': int(np.sum((abs(upper_triangle) >= 0.3) & (abs(upper_triangle) < 0.7))),
            'high': int(np.sum((abs(upper_triangle) >= 0.7) & (abs(upper_triangle) < 0.9))),
            'very_high': int(np.sum(abs(upper_triangle) >= 0.9))
        }
    }
    
    if len(high_corr_pairs) > 0:
        correlation_stats['top_correlated_pairs'] = sorted(
            high_corr_pairs, key=lambda x: x['correlation'], reverse=True
        )[:20]
    
    logger.info(f"Paires hautement corrélées (>{threshold}): {len(high_corr_pairs)}")
    logger.info(f"Corrélation moyenne: {correlation_stats['mean_correlation']:.4f}")
    
    return correlation_stats


if __name__ == '__main__':
    # Test du module
    print("🔧 Test du module d'ingénierie des features EEG")
    
    # Générer des données de test
    np.random.seed(42)
    n_samples, n_features = 1000, 100
    X_test = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y_test = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    print(f"Dataset de test: {X_test.shape}")
    
    # Test de sélection rapide
    X_selected = quick_feature_selection(X_test, y_test, method='importance', n_features=20)
    print(f"Features sélectionnées: {X_selected.shape}")
    
    # Test d'analyse de corrélation
    corr_analysis = analyze_feature_correlation(X_test, threshold=0.8)
    print(f"Analyse de corrélation: {corr_analysis['high_correlation_pairs']} paires hautement corrélées")
    
    print("✅ Tests terminés avec succès!")