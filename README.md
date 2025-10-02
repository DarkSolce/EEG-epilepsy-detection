# 🧠 EpiDetect AI - Epilepsy Seizure Detection System

## 🌟 Overview
**EpiDetect AI** is an advanced machine learning system designed to assist in the detection of epileptic seizures through automated analysis of EEG (Electroencephalogram) signals. This project combines signal processing, feature engineering, and ensemble machine learning to provide accurate seizure detection with a user-friendly web interface.

![Streamlit App](https://img.shields.io/badge/Web%20App-Streamlit-%23FF4B4B?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-94.9%25-brightgreen?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=for-the-badge)

## 🎯 Key Features
- **🤖 Multi-Model AI Analysis**: Three ensemble models (Random Forest, Gradient Boosting, Logistic Regression)
- **📊 Real-time Visualization**: Interactive dashboards with Plotly charts and gauges
- **⚡ High Performance**: 94.9% accuracy on balanced EEG datasets
- **🎨 Modern Web Interface**: Dark-themed Streamlit application with responsive design
- **🔬 Advanced Signal Processing**: 2,000+ features extracted from raw EEG data

## 🏗️ Project Structure
EEG-epilepsy-detection/
├── 📁 Data/
│ ├── Raw/ # Original EEG datasets
│ ├── processed/ # Processed features and scalers
│ └── analysis/ # Data analysis reports
├── 📁 models/ # Trained machine learning models
├── 📁 notebooks/ # Jupyter notebooks for exploration
├── 📁 utils/ # Utility functions and helpers
├── 🐍 app.py # Main Streamlit application
├── 🔧 retrain_clean.py # Model training pipeline
├── 📊 requirements.txt # Python dependencies
└── 📖 README.md # Project documentation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/DarkSolce/EEG-epilepsy-detection.git
cd EEG-epilepsy-detection

# Install dependencies
pip install -r requirements.txt

# Launch the web application
streamlit run app.py
📊 Dataset Information
Sources
Temple University EEG Corpus (TUEG)

CHB-MIT Scalp EEG Database

Preprocessing Pipeline
Signal Cleaning: Noise removal and artifact handling

Feature Extraction: 2,000+ temporal and spectral features

Data Balancing: SMOTE + Random UnderSampling

Feature Selection: 36,865 → 2,000 most important features

🛠️ Technical Stack
Core Technologies
Python 3.8+: Primary programming language

Scikit-learn: Machine learning models and preprocessing

Streamlit: Web application framework

Plotly: Interactive visualizations

Machine Learning
Random Forest: Ensemble tree-based classifier

Gradient Boosting: Sequential model optimization

Logistic Regression: Baseline linear model

Feature Engineering: Comprehensive signal processing

Data Processing
Pandas & NumPy: Data manipulation

MNE-Python: EEG signal processing

Joblib: Model serialization

📈 Model Performance
Model	Accuracy	Precision	Recall	F1-Score
Random Forest	94.9%	95.2%	94.7%	94.9%
Gradient Boosting	93.8%	94.1%	93.5%	93.8%
Logistic Regression	91.2%	91.5%	90.8%	91.1%
💻 Usage
Web Application
bash
streamlit run app.py
The application provides:

Real-time Predictions: Upload EEG data for instant analysis

Model Comparison: Performance metrics across all models

Interactive Visualizations: Confusion matrices, probability gauges

Data Export: Download results in CSV format

Programmatic Usage
python
from utils.preprocessing import load_and_preprocess
from models.predict import make_prediction

# Load and preprocess data
data = load_and_preprocess('path/to/eeg_data.csv')

# Make predictions
predictions, probabilities = make_prediction(data)
🔧 Training Pipeline
To retrain models with new data:

bash
python retrain_clean.py
This executes:

Data loading and preprocessing

Feature selection and engineering

Model training with cross-validation

Performance evaluation

Model serialization

📱 Application Features
🎯 Prediction Interface
File upload for EEG data

Real-time analysis with progress indicators

Probability gauges and confidence scores

Batch processing for multiple recordings

📊 Analytics Dashboard
Model performance comparison

Interactive confusion matrices

Feature importance visualizations

Statistical analysis reports

🎨 User Experience
Dark theme optimized for medical use

Responsive design for various screen sizes

Accessibility considerations

Multi-language support ready

⚠️ Important Disclaimer
EpiDetect AI is a decision support tool and NOT a replacement for medical diagnosis.

🔬 Always consult qualified neurologists for medical decisions

🏥 Use in conjunction with professional medical assessment

📋 Results should be interpreted by healthcare professionals

⚠️ Not certified for standalone clinical use

🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Developer
Skander Chebbi

Data Scientist & Machine Learning Engineer

Specialized in Healthcare AI and Signal Processing

GitHub Profile

🙏 Acknowledgments
Temple University for the TUEG dataset

CHB-MIT for the scalp EEG database

The open-source community for invaluable tools and libraries
