# FitbitAnalysis
Analysis of Fitbit activity and sleep data to extract insights on user behavior, sleep patterns, and daily activity using data analytics and visualization techniques.
# Multivariate Clustering Analysis on Fitbit and Mobile Usage Data

This project applies clustering techniques to analyze data collected from:
- Fitbit smartwatch (sleep stages, activity, calories burned, stress)
- Smartphone usage (screen time, notifications, app openings)

Objectives
- Discover hidden patterns in user behavior using clustering.
- Evaluate cluster quality using the Silhouette Score.
- Apply and compare different clustering algorithms.

Algorithms Used
- K-Means (k=2 to 10)
- DBSCAN (grid search on eps and min_samples)
- Mean Shift
- Hierarchical Clustering (Ward linkage)

Datasets and Features
- **Sleep-related:** Sleep_Stage, dateOfSleep, minutesToFallAsleep, minutesAsleep, timeInBed, SleepScore
- **Activity-related:** CaloriesBurned, Steps, Distance km, stress_1, stress_2, stress_3
- **Mobile usage:** Notifications, Screen Time (per hour and total), App Openings

#Highlights
- K-Means: Best k=2 for sleep, k=10 for steps & mobile use
- DBSCAN: Silhouette scores up to 1.00, 0% outliers – highly successful
- Mean Shift: Mixed results, suitable for activity features
- Hierarchical Clustering: Found strong binary and multi-cluster structures

Project Contents
- `FinalDataset.pdf` – Full academic report (English)
- `FinalDataset.docx` – Editable version
- `plots/` – Visual clustering results (optional folder)
- `FinalDataset.csv` – Raw or preprocessed dataset (optional)

#Tools Used
- Python, Scikit-learn, Pandas, Matplotlib, t-SNE

This project was developed by **Nikolaos Marios Fountas** as part of Erasmus+ academic work (2025) and demonstrates applied knowledge in unsupervised learning and data-driven insight extraction.
