# House Price Prediction using PyTorch

An end-to-end deep learning pipeline designed to predict tabular house prices using a fully connected Deep Neural Network (DNN) implemented in PyTorch.

## 🚀 Key Features
* **Advanced Data Preprocessing:** One-hot encoding for categorical variables and feature scaling using `StandardScaler` from scikit-learn.
* **Robust Architecture:** Built with multiple linear layers, **Batch Normalization** for faster convergence, and **Dropout Regularization** to prevent overfitting.
* **Hardware Acceleration:** Seamless device targeting with automatic CUDA/GPU utilization.
* **Comprehensive Evaluation:** Tracked using R² Score, MAE, MSE, and RMSE on original-scale (inverse-transformed) data.

## 🛠️ Tech Stack
* Python
* PyTorch (`torch.nn`)
* Pandas & NumPy
* Scikit-Learn
* Matplotlib (for loss curves)
