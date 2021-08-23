import pandas as pd
import numpy as np

df = pd.read_csv("origin_data/Grid_Disruption_00_14_standardized - Grid_Disruption_00_14_standardized.csv")
print(df.head())


missing_values_count = df.isnull().sum()
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()
print("\nAnalyse: % fehlende Daten:")
print((total_missing/total_cells) * 100)

print(df["Time Event Began"].dtype)

print(df['Time Event Began'].unique())