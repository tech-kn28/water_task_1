# Week-1  
This machine learning project focuses on predicting whether water is safe for consumption by analyzing various chemical parameters. Developed during the Shell-Edunet Skills4Future Internship (June–July 2025), it marks my first hands-on experience in AI/ML and represents a meaningful step toward sustainable, data-driven solutions.

# 🌊 Water Quality Prediction Using Machine Learning

> **“Turning chemical signals into clarity — one prediction at a time.”**

---

## 📌 Project Overview

This initiative, completed under the Edunet Foundation AI-ML Internship, utilizes historical water sample data collected from Punjab (2000–2021) to predict critical water quality indicators. By applying machine learning techniques, the model provides estimations for pollutant levels, supporting safer water management and environmental protection efforts.

As clean water becomes increasingly vital due to rising pollution and climate change, predictive modeling helps enable early warnings and promotes long-term ecological sustainability.

---

## ⚙️ Technologies Used

- **Python** — Primary language for development  
- **Pandas** — Data manipulation and cleaning  
- **NumPy** — Efficient numerical computation  
- **Matplotlib & Seaborn** — Visual data exploration  
- **Scikit-learn** — Core ML algorithms and evaluation tools

---

## 💧 Predicted Water Quality Parameters

The model estimates the concentrations of key chemical substances such as:

- **O₂** (Dissolved Oxygen)  
- **NO₃** (Nitrates)  
- **NO₂** (Nitrites)  
- **SO₄** (Sulfates)  
- **PO₄** (Phosphates)  
- **Cl⁻** (Chlorides)

These indicators are essential in assessing the usability of water for human consumption, agriculture, and aquatic ecosystems.

---

## 🤖 Model & Methodology

- **Model Used**: A `RandomForestRegressor` wrapped inside `MultiOutputRegressor` to predict multiple parameters simultaneously.  
- **Features Used**:  
  - NH₄ (Ammonium)  
  - BSK5 (5-day Biochemical Oxygen Demand)  
  - Suspended Solids  
  - Year  
  - Month  

- **Missing Data Handling**: Median-based imputation  
- **Data Split Ratio**: 80% for training, 20% for testing  
- **Evaluation Metrics**:  
  - R² Score — measures model fit  
  - Mean Squared Error — quantifies prediction error

---

## 📊 Model Performance Snapshot

| Parameter | R² Score | Mean Squared Error |
|-----------|----------|--------------------|
| O₂        | 0.82     | 3.12               |
| NO₃       | 0.85     | 2.45               |
| NO₂       | 0.78     | 0.09               |
| SO₄       | 0.87     | 120.55             |
| PO₄       | 0.80     | 0.07               |
| CL        | 0.91     | 140.34             |

> *(Note: These values are sample placeholders — replace with actual evaluation results.)*

---

## ✅ Final Notes

This project demonstrates how machine learning can be effectively applied to monitor environmental quality, particularly in water resource management. With additional features like rainfall data, land use patterns, or industrial mapping, this model can be scaled for broader deployment and better accuracy in real-time applications.

---

**Built by Krishnnarayan  **
