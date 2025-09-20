#!/usr/bin/env python3
"""
Main Execution Script for EEG Epilepsy Detection Pipeline
=========================================================

This script provides a command-line interface to run the complete
EEG epilepsy detection pipeline with various options and configurations.

Usage:
    python main.py --data data/raw/EEG_Scaled_data.csv --quick
    python main.py --data your_data.csv --models "Random Forest,Logistic Regression"
    python main.py --config config.json

Author: Skander Chebbi
Date: 2025
"""

import argparse
import json
import sys
import os
from pathlib import Path
import logging

# Ajouter le répertoire src au path pour les imports
sys.path.append(str(Path(__file__).parent / "src"))

# Imports du package
try:
    from src import (
        create_complete_pipeline, 
        quick_start, 
        DEFAULT_CONFIG, 
        get_package_info,
        print_welcome
    )
    PACKAGE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous que tous les modules dans src/ sont correctement installés")
    PACKAGE_AVAILABLE = False


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configurer le système de logging"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Format des messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    handlers = [console_handler]
    
    # Handler pour fichier si spécifié
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configuration du logger principal
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)


def validate_data_file(data_path: str) -> bool:
    """Valider que le fichier de données existe et est accessible"""
    if not os.path.exists(data_path):
        print(f"❌ Fichier de données non trouvé: {data_path}")
        return False
    
    if not data_path.endswith(('.csv', '.edf', '.mat')):
        print(f"⚠️  Format de fichier non standard: {data_path}")
        print("Formats recommandés: .csv, .edf, .mat")
    
    # Vérifier la taille du fichier
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    print(f"📁 Fichier de données: {data_path} ({file_size_mb:.1f} MB)")
    
    return True


def create_custom_config(args) -> dict:
    """Créer une configuration personnalisée basée sur les arguments"""
    config = DEFAULT_CONFIG.copy()
    
    # Configuration des modèles
    if args.models:
        models_list = [m.strip() for m in args.models.split(',')]
        config['modeling']['models'] = models_list
        print(f"🤖 Modèles sélectionnés: {models_list}")
    
    # Mode rapide
    if args.quick:
        config['modeling']['quick_training'] = True
        config['modeling']['models'] = ['Random Forest', 'Logistic Regression']
        config['feature_engineering']['max_features'] = 1000
        config['modeling']['cv_folds'] = 3
        print("⚡ Mode rapide activé")
    
    # Gestion du déséquilibre
    if hasattr(args, 'no_balance') and args.no_balance:
        config['modeling']['handle_imbalance'] = False
        print("⚖️ Gestion du déséquilibre désactivée")
    
    # Répertoires de sortie personnalisés
    if args.output_dir:
        config['output']['models_dir'] = os.path.join(args.output_dir, 'models')
        config['output']['results_dir'] = os.path.join(args.output_dir, 'results')
        config['data']['processed_data_dir'] = os.path.join(args.output_dir, 'processed')
        print(f"📁 Répertoire de sortie: {args.output_dir}")
    
    return config


def load_config_file(config_path: str) -> dict:
    """Charger une configuration depuis un fichier JSON"""
    try:
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
        
        # Fusionner avec la configuration par défaut
        config = DEFAULT_CONFIG.copy()
        
        # Mise à jour récursive des dictionnaires imbriqués
        def update_nested_dict(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_nested_dict(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        update_nested_dict(config, custom_config)
        print(f"📋 Configuration chargée depuis: {config_path}")
        return config
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la configuration: {e}")
        print("Utilisation de la configuration par défaut")
        return DEFAULT_CONFIG.copy()


def save_config_template(output_path: str = "config_template.json"):
    """Sauvegarder un template de configuration"""
    try:
        with open(output_path, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"✅ Template de configuration sauvegardé: {output_path}")
        print("Vous pouvez le modifier et l'utiliser avec --config")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du template: {e}")


def print_results_summary(results: dict):
    """Afficher un résumé des résultats"""
    if not results.get('pipeline_info', {}).get('success', False):
        print(f"❌ Pipeline échoué: {results.get('pipeline_info', {}).get('error', 'Erreur inconnue')}")
        return
    
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print("="*70)
    
    # Performance du meilleur modèle
    best_model = results['final_performance']['best_model']
    best_metrics = results['final_performance']['best_metrics']
    
    print(f"🏆 Meilleur modèle: {best_model}")
    print(f"📈 Performances:")
    print(f"   • Accuracy:    {best_metrics['accuracy']:.4f}")
    print(f"   • Precision:   {best_metrics['precision']:.4f}")
    print(f"   • Recall:      {best_metrics['recall']:.4f}")
    print(f"   • F1-Score:    {best_metrics['f1_score']:.4f}")
    print(f"   • Specificity: {best_metrics['specificity']:.4f}")
    
    if 'roc_auc' in best_metrics:
        print(f"   • ROC-AUC:     {best_metrics['roc_auc']:.4f}")
    
    # Informations sur les données
    preprocessing = results['preprocessing_report']
    feature_eng = results['feature_engineering_report']
    
    print(f"\n🔢 Données:")
    print(f"   • Features originales: {preprocessing['original_shape'][1]:,}")
    print(f"   • Features finales:    {feature_eng['final_features']:,}")
    print(f"   • Réduction:           {feature_eng['reduction_ratio']:.1%}")
    print(f"   • Échantillons train:  {preprocessing['train_shape'][0]:,}")
    print(f"   • Échantillons test:   {preprocessing['test_shape'][0]:,}")
    
    # Comparaison des modèles
    print(f"\n🤖 Comparaison des modèles:")
    comparison = results['final_performance']['model_comparison']
    for model_result in comparison[:3]:  # Top 3
        name = model_result['Model']
        f1 = model_result['F1-Score']
        print(f"   • {name:<20}: F1 = {f1:.4f}")
    
    print("\n" + "="*70)


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Pipeline de détection d'épilepsie basée sur l'EEG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s --data data/raw/EEG_data.csv --quick
  %(prog)s --data my_data.csv --models "Random Forest,SVM"
  %(prog)s --config my_config.json
  %(prog)s --create-config-template
  %(prog)s --info
        """
    )
    
    # Arguments principaux
    parser.add_argument('--data', type=str, 
                       help='Chemin vers le fichier de données EEG')
    
    parser.add_argument('--config', type=str,
                       help='Chemin vers le fichier de configuration JSON')
    
    # Options de modélisation
    parser.add_argument('--models', type=str,
                       help='Modèles à entraîner (séparés par des virgules)')
    
    parser.add_argument('--quick', action='store_true',
                       help='Mode rapide (paramètres optimisés pour la vitesse)')
    
    parser.add_argument('--no-balance', action='store_true',
                       help='Désactiver la gestion du déséquilibre des classes')
    
    # Options de sortie
    parser.add_argument('--output-dir', type=str,
                       help='Répertoire de sortie personnalisé')
    
    parser.add_argument('--log-file', type=str, default='logs/pipeline.log',
                       help='Fichier de log (défaut: logs/pipeline.log)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Affichage détaillé (niveau DEBUG)')
    
    # Utilitaires
    parser.add_argument('--create-config-template', action='store_true',
                       help='Créer un template de configuration')
    
    parser.add_argument('--info', action='store_true',
                       help='Afficher les informations sur le package')
    
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    # Vérifier la disponibilité du package
    if not PACKAGE_AVAILABLE:
        print("❌ Package non disponible. Vérifiez l'installation.")
        sys.exit(1)
    
    # Configuration du logging
    logger = setup_logging(args.verbose, args.log_file)
    
    # Traitement des commandes utilitaires
    if args.info:
        print_welcome()
        package_info = get_package_info()
        print(f"Version: {package_info['version']}")
        print(f"Modules: {package_info['main_modules']}")
        return
    
    if args.create_config_template:
        save_config_template()
        return
    
    # Validation des arguments
    if not args.data and not args.config:
        print("❌ Vous devez spécifier --data ou --config")
        parser.print_help()
        sys.exit(1)
    
    # Déterminer le chemin des données
    if args.config:
        config = load_config_file(args.config)
        data_path = config['data']['raw_data_path']
    else:
        data_path = args.data
        config = create_custom_config(args)
    
    # Valider le fichier de données
    if not validate_data_file(data_path):
        sys.exit(1)
    
    # Afficher la bienvenue et les paramètres
    print_welcome()
    print(f"📁 Données: {data_path}")
    print(f"🤖 Modèles: {config['modeling']['models']}")
    print(f"⚡ Mode rapide: {'Oui' if config['modeling']['quick_training'] else 'Non'}")
    print()
    
    try:
        # Exécution du pipeline
        logger.info("Démarrage du pipeline principal")
        
        if args.quick and not args.config:
            # Utiliser le démarrage rapide
            results = quick_start(data_path)
        else:
            # Pipeline complet avec configuration
            results = create_complete_pipeline(data_path, config)
        
        # Afficher les résultats
        print_results_summary(results)
        
        # Sauvegarder un résumé simple
        summary_path = os.path.join(config['output']['results_dir'], 'pipeline_summary.txt')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            f.write("=== RÉSUMÉ DU PIPELINE EEG EPILEPSIE ===\n")
            f.write(f"Date: {results['pipeline_info'].get('completion_time', 'N/A')}\n")
            f.write(f"Données: {data_path}\n")
            f.write(f"Meilleur modèle: {results['final_performance']['best_model']}\n")
            f.write(f"F1-Score: {results['final_performance']['best_metrics']['f1_score']:.4f}\n")
        
        print(f"📝 Résumé sauvegardé: {summary_path}")
        logger.info("Pipeline terminé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}")
        print(f"❌ Erreur: {str(e)}")
        print("Consultez le fichier de log pour plus de détails")
        sys.exit(1)


if __name__ == '__main__':
    main()