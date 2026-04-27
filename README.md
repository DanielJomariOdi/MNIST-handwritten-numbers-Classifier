## Overview
This repository contains a complete machine learning pipeline for classifying handwritten digits. The project explores various traditional machine learning algorithms and combines them into a robust ensemble model. It handles everything from data preprocessing and dimensionality reduction to hyperparameter tuning and model deployment.

## Workflow & Pipeline
1. **Data Preprocessing**: Raw pixel intensities are normalized using `StandardScaler` to ensure all features contribute equally to the distance calculations.
2. **Dimensionality Reduction**: Principal Component Analysis (`PCA`) is applied to reduce the feature space from 784 pixels down to 234 principal components. This retains 90% of the original variance while significantly speeding up training time and reducing noise.
3. **Model Training**: Multiple classification pipelines are built. Hyperparameters for each model are systematically optimized using `GridSearchCV`.
4. **Ensemble Learning**: A `VotingClassifier` aggregates the predictions of the individual base models to improve overall accuracy and generalization.

## Models Evaluated
- **K-Nearest Neighbors (KNN)**: A distance-based baseline classifier that groups similar pixel patterns.
- **Support Vector Machine (SVM)**: A margin-based classifier highly effective at finding optimal boundaries in high-dimensional spaces.
- **Random Forest**: A tree-based ensemble method used to handle non-linearities and prevent overfitting.
- **Voting Classifier (Ensemble)**: The ultimate best-performing model that combines the strengths of the algorithms above to correct individual model biases.

## Evaluation Metrics
The models are evaluated using:
- **Classification Report**: Precision, Recall, and F1-scores across all 10 digit classes.
- **Confusion Matrix**: A detailed visualization to analyze specific misclassifications (e.g., distinguishing between visually similar digits like '4' and '9' or '3' and '8').

## Installation & Requirements
To run the notebook locally, ensure you have Python installed along with the following libraries:

```
```text?code_stdout&code_event_index=2
Successfully generated README.md
[file-tag: README.md]

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn joblib
```

## Usage
1. Clone the repository and navigate to the directory.
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Jupyther Notebook.ipynb"
   ```
3. Run the cells sequentially to load the data, train the models, and visualize the performance metrics.
4. The final phase of the notebook will export the best models for production.

## Saved Models / Artifacts
Upon successful execution of the notebook, the following serialized files will be generated using `joblib` for easy deployment:
- `mnist_ensemble.pkl`: The final Voting Classifier model.
- `mnist_svm.pkl`: The trained SVM model.
- `mnist_knn.pkl`: The trained KNN model.
- `mnist_scaler.pkl`: The fitted StandardScaler.
- `mnist_pca.pkl`: The fitted PCA transformer.
"""
