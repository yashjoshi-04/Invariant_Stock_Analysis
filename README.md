# Adapting the InvariantStock Framework for Enhanced Stock Return Prediction

This repository presents the implementation of the **InvariantStock Framework** to improve stock return predictions, especially during market turbulence. The project integrates environment-agnostic and environment-aware features into traditional models like the **Fama-French 3-Factor Model**, demonstrating robustness and adaptability across shifting market conditions.

---

## 🚀 **Key Objectives**
1. Enhance stock return prediction accuracy using invariant features.
2. Integrate environment-dependent metrics for context-specific adaptability.
3. Address limitations of traditional models during significant market events (e.g., COVID-19, GFC).

---

## 📋 **Methodology**
### **Framework Overview**
1. **Environment-Agnostic Features**:
   - Metrics capturing intrinsic stock/company attributes.
   - Examples: Price-to-Earnings (P/E), Price-to-Book (P/B), Market Capitalization.

2. **Environment-Dependent Features**:
   - Metrics reflecting market-specific dynamics.
   - Examples: Price, Volume, and Percentage Changes.

### **Data Sources**
- **Primary**: WRDS, Yahoo! Finance, Bloomberg, Kenneth French Data Library.
- **Supplementary**: Federal Reserve Economic Data (FRED), macroeconomic indicators (interest rates, inflation, GDP growth).

### **Data Preprocessing**
- Stationarity Testing: KPSS and ADF tests.
- Time Series Decomposition: Analysis of Autocorrelation (ACF) and Partial Autocorrelation (PACF).
- Data Splitting: 80-20 training/testing split.
- Dummy Variables: Inclusion of market events like COVID-19 and GFC.

### **Models Used**
- Traditional Models: OLS, ARIMA, ARIMA-X.
- Advanced Machine Learning: Elastic Net, Random Forest Regressor, Gradient Boosting, LightGBM.
- 3D ML Framework: Implemented using PyTorch for invariant feature extraction.

---

## 📊 **Results**
### **Performance Highlights**
1. **Initial Model**:
   - Significant variables: Limited.
   - Challenges: Multicollinearity, inability to predict downturns accurately.

2. **Improved Model (PCA)**:
   - Resolved multicollinearity.
   - Persistent Issues: Increased RMSE and limited downturn prediction.

3. **Final Model (Invariant Framework)**:
   - Integrated 3D ML methods for stable, environment-agnostic features.
   - Key Achievements:
     - Improved prediction accuracy for market downturns.
     - Reduced RMSE significantly.

---

## 🔍 **Key Insights**
- Traditional time-series models often fail to adapt to nonlinear and dynamic market changes.
- The **InvariantStock Framework** enhances predictive stability by focusing on robust feature selection and environment-aware adaptability.

---

## 🌟 **Future Extensions**
- Incorporate regime-switching models for dynamic market phases.
- Explore additional economic and technical indicators for a more comprehensive model.
- Demonstrate scalability with models like **LightGBM** to improve real-time prediction and strategy implementation.

---

## 🛠 **Technologies Used**
- Python: For data processing, model training, and visualization.
- PyTorch: To implement 3D ML models.
- Libraries: Scikit-learn, NumPy, Pandas, Statsmodels.

---

## 📜 **References**
1. Cao, H., Zou, J., Liu, Y., et al. (2024). [InvariantStock: Learning invariant features for mastering the shifting market](https://openreview.net/pdf?id=dtNEvUOZmA). *Transactions on Machine Learning Research*.

---



For inquiries, contact [Yash Joshi](yashak.j.2024@mqf.smu.edu.sg).

---

## 📈 **Future Goals**
- Enhance model adaptability with real-time datasets.
- Deploy the framework in live trading environments.

---
