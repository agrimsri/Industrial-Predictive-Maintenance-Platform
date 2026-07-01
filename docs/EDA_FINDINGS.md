# C-MAPSS Exploratory Data Analysis (EDA) Findings

This document summarizes the insights gained from the EDA performed on the NASA C-MAPSS dataset ([`01_eda_cmapss.ipynb`](file:///home/silvanus/CODES/ipmp-platform/ml/notebooks/01_eda_cmapss.ipynb)). These findings justify the design choices and feature engineering steps implemented in the modeling phase.

## 1. Constant/Flat Sensors (FD001)
In the FD001 dataset, several sensors show zero or near-zero variance across all engine units and cycles. These sensors do not change over time and carry no predictive power for RUL estimation.
- **Sensors to drop:** `s_1`, `s_5`, `s_6`, `s_10`, `s_16`, `s_18`, `s_19` (often standard deviations are less than `1e-4`).
- **Justification:** Dropping these features reduces input dimensionality, eliminates noise, and prevents numerical instabilities in modeling.

## 2. Sensor Correlations with RUL
Computing the Pearson correlation coefficient between each active sensor and the target RUL reveals distinct patterns:
- **Strong Positive Correlation (increases as RUL increases / decreases over time as engine degrades):**
  - `s_11` (HPC outlet temperature)
  - `s_12` (LPT outlet pressure)
  - `s_7` (HPC outlet pressure)
- **Strong Negative Correlation (decreases as RUL increases / increases over time as engine degrades):**
  - `s_2` (LPC outlet temperature)
  - `s_3` (HPC outlet temperature)
  - `s_4` (LPT outlet temperature)
  - `s_15` (LPT outlet temperature)
  - `s_21` (Engine physical fan speed)
- **Weakly Correlated Sensors:** Sensors like `s_9` and `s_14` show moderate correlations, while others exhibit non-linear degradation shapes.

## 3. Operational Regimes (FD002 & FD004)
Unlike FD001 which has a single operating condition, FD002 and FD004 are simulated under **6 operating conditions** characterized by:
- Altitude
- Mach number
- Throttle resolver angle

### K-Means Clustering on Settings
Applying K-Means clustering on the 3 operational settings (`setting_1`, `setting_2`, `setting_3`) partitions the dataset into exactly 6 clusters.
- **Impact on Modeling:** If we do not normalize the sensor values per regime, the scale shifts will overwhelm the degradation signals. For example, the mean value of `s_2` shifts significantly across different operating regimes.
- **Strategy:** In Milestone 1.2, we must normalize sensor readings *within each operating regime* (z-score normalization per cluster) or pass the cluster/regime ID as a categorical feature to the model.
