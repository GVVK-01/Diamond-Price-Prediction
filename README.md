# Diamond-Price-Prediction
# Diamond Price Prediction using Machine Learning

This project predicts the price of diamonds based on their physical and categorical attributes. The model was trained using regression techniques and evaluated using RMSE and R² scores. Multiple machine learning models were compared and optimized to identify the best-performing algorithm for this prediction task.

---

## 🧾 Dataset Description

The dataset contains 53,940 diamonds with attributes:

- **carat**: weight of the diamond
- **cut**: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **color**: diamond color quality (D best → J worst)
- **clarity**: measures inclusions (I1 worst → IF best)
- **depth**: depth percentage
- **table**: width percentage
- **x (length)**, **y (width)**, **z (depth_mm)**
- **price**: final selling price (USD)

---

## 🧩 Problem Statement

> Predict the final selling price of diamonds using their physical and grading attributes.

Diamond pricing is non-linear and influenced by multiple interacting factors. Machine learning helps estimate pricing more accurately than linear statistics.

---

## 📊 Exploratory Data Analysis (EDA)

Key EDA findings:

- Carat has the strongest correlation with price.
- Higher clarity, cut, and color grades correspond to higher prices.
- Price distribution is right-skewed.
- Rare outliers correspond to very expensive diamonds (valid points).
- Zero values in x,y,z dimensions were treated as invalid entries and removed.

Visualizations included:
- Histograms
- Boxplots
- Scatter plots
- Correlation heatmaps
- Pairplots

---

## 🏗 Preprocessing Steps
 Duplicates removed  
Invalid dimension outliers removed  
Ordinal Encoding for:
  - cut
  - color
  - clarity
Train-test split (80/20 ratio)  
Scaling applied **only to Linear Regression & KNN**  
Tree models trained without scaling  

Reasoning:
- Tree-based models are scale invariant
- Distance/gradient-based models require scaling

---

## 🤖 Model Development

The following regression models were trained:

1. Linear Regression  
2. KNN Regression  
3. Decision Tree Regressor  
4. Random Forest Regressor  
5. Gradient Boosting Regressor  

Models were evaluated using:

- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)
- Train/Test comparison for generalization

---

## 📈 Model Results

| Model | Train RMSE | Test RMSE | Train R² | Test R² | Notes |
|---|---|---|---|---|---|
| Linear Regression | 1188 | 1194 | 0.881 | 0.876 | Underfit |
| KNN | 598 | 760 | 0.970 | 0.950 | Good |
| Decision Tree | 6 | 607 | 1.000 | 0.968 | Overfit |
| **Random Forest** | **169** | **450** | **0.998** | **0.982** | **Best** |
| Gradient Boosting | 532 | 550 | 0.976 | 0.974 | Very Good |

---

## 🏆 Best Model Selection

The **Random Forest Regressor** was selected as the best model due to:
 Lowest Test RMSE  
Highest Test R²  
Strong generalization  
Non-linear pattern capturing  
Handles ordinal + numeric features effectively  

---

## 🛠 Hyperparameter Tuning (GridSearchCV)

**Grid Search Best Parameters:**

{'n_estimators': 300,
'max_depth': None,
'max_features': 'sqrt',
'min_samples_leaf': 1,
'min_samples_split': 2}
**Tuned Performance:**

| Metric | Value |
|---|---|
| Train RMSE | 184.68 |
| Test RMSE | 493.13 |
| Train R² | 0.997 |
| Test R² | 0.979 |

Tuning did not improve the model further, so the **base Random Forest** was selected as the final model.

---

## 📤 Final Conclusion

The **Random Forest Regressor (untuned)** demonstrated the best predictive performance and generalization and was selected as the final model. This indicates that tree-based ensemble methods are well-suited for diamond price prediction due to their ability to model non-linear relationships between diamond attributes.

---

## Tech Stack
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost (optional)
