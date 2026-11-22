# 🚕 ML Model Reliability Under Data Drift (Self-Healing ML Pipeline)

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Dashboard-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/fatimajaffar/Project)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-orange)](https://xgboost.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)](https://www.gradio.app/)

An end-to-end automated data drift monitoring and self-healing retraining pipeline built on **150,000 NYC Yellow Taxi trip records** across three distinct economic eras (2019, 2020, and 2023). The system distinguishes harmless **Covariate Shift** from harmful **Concept Drift** using statistical tests (PSI & KS-Test) and triggers chronological adaptive retraining only when reliability degrades.

🔗 **Live Interactive Web App:** [https://huggingface.co/spaces/fatimajaffar/Project](https://huggingface.co/spaces/fatimajaffar/Project)  
🔗 **Direct Full-Screen App:** [https://fatimajaffar-project.hf.space](https://fatimajaffar-project.hf.space)

---

## 📌 Problem Statement
In production environments, machine learning models silently decay as real-world data distributions evolve:
- **Blind periodic retraining** wastes compute resources when underlying feature-target relationships have not changed.
- **Static deployment (never retraining)** leads to severe prediction errors during structural economic or behavioral shifts.

This project implements a **drift-triggered self-healing controller** that monitors both unsupervised prediction drift ($\hat{y}$ PSI) and supervised target/feature drift ($X, y$ PSI & KS-Test) to retrain models only when necessary.

---

## 📊 Dataset & Experimental Eras
- **Source:** NYC Yellow Taxi Trip Records (50,000 cleaned, schema-validated records per era; 150,000 total).
- **Target Variable:** `fare_amount` (USD).
- **Engineered Features:** `passenger_count`, `trip_distance`, `ratecodeid`, `pulocationid`, `dolocationid`, `payment_type`, `trip_duration_min`, `pickup_hour`, `day_of_week`, and `is_weekend`.

| Era | Real-World Event | Drift Regime | Observed Behavior |
| :--- | :--- | :--- | :--- |
| **2019 (Baseline)** | Pre-pandemic normal operations | Reference Distribution | Baseline XGBoost trained with chronological 80/20 split ($R^2 > 0.96$). |
| **2020 (T1 Batch)** | COVID-19 Lockdown | **Covariate Shift** (Input $X$ shift) | Trip distances shifted, but fare rules stayed intact — static model stayed reliable ($R^2 > 0.95$). |
| **2023 (T2 Batch)** | Post-COVID Fare Inflation | **Concept Drift** (Target $y$ shift) | Base fare & surcharge hikes caused severe drift ($PSI = 0.8162$), degrading static accuracy ($R^2 \approx 0.82$). |

---

## ⚙️ System Architecture (6-Stage Pipeline)
1. **Data Ingestion & Sanitization:** Removed missing values, zero/negative fares, impossible durations, and extreme sensor outliers across all three eras.
2. **Temporal Feature Engineering:** Extracted `trip_duration_min`, `pickup_hour`, `day_of_week`, and `is_weekend` to capture rush-hour and weekend pricing patterns.
3. **Chronological Baseline Training:** Trained `XGBRegressor` strictly on the first 80% of 2019 data and validated on the remaining 20% holdout window to prevent temporal leakage.
4. **Dual-Layer Statistical Drift Detection:**
   - **Population Stability Index (PSI):** Quantifies distribution shift across input features ($X$), predicted outputs ($\hat{y}$), and target values ($y$) using standard thresholds ($<0.10$ Safe, $0.10\text{–}0.20$ Moderate, $\ge 0.20$ Severe).
   - **Kolmogorov-Smirnov (KS) Two-Sample Test:** Measures nonparametric distributional divergence and statistical significance ($p$-values).
5. **Chronological Adaptive Retraining:** When drift and error thresholds are breached, the controller adapts the model on the first 50% of the new production era and evaluates recovery on the unseen second 50% holdout window.
6. **Cloud Deployment & Custom CSV Auditing:** Deployed via Gradio on Hugging Face Spaces with live benchmark visualizations and a custom CSV uploader for auditing any tabular regression dataset.

---

## 📈 Key Results & Takeaways
- **Avoiding Unnecessary Retraining (2020):** Despite noticeable input covariate drift during the 2020 lockdown, static XGBoost maintained $R^2 > 0.95$, proving that input shift alone does not justify retraining costs.
- **Self-Healing Recovery (2023):** Detecting the 2023 fare inflation concept drift and triggering adaptive retraining restored holdout accuracy from $R^2 \approx 0.82$ back to **$R^2 \approx 0.98$** while achieving up to **75% simulated compute reduction** compared to blind periodic retraining across evaluation windows.

---

## ⚠️ Methodological Notes & Limitations
- **Post-Trip Feature Context:** `trip_duration_min` is derived from actual trip timestamps, making this pipeline suited for post-trip fare auditing and dynamic pricing calibration rather than pre-trip dispatch estimation.
- **Label-Delayed Environments:** Because ground-truth labels ($y$) arrive with latency in real production systems, the dashboard tracks **unsupervised prediction drift ($\hat{y}$ PSI)** alongside supervised target metrics.

---

## 🚀 Run Locally
```bash
git clone [https://github.com/Fatima0313/ML-Model-Reliability-Data-Drift.git](https://github.com/Fatima0313/ML-Model-Reliability-Data-Drift.git)
cd ML-Model-Reliability-Data-Drift
pip install -r requirements.txt
python app.py
