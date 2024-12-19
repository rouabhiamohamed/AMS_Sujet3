import pandas as pd

# Chemins vers vos fichiers
file1_path = 'data/my_submission.csv'
file2_path = 'data/my_submission(5).csv'

# Charger les fichiers
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Comparer les différences
comparison = df1.compare(df2)

# Exporter les différences
differences_output_path = 'data/differences_graphml.csv'
comparison.reset_index().to_csv(differences_output_path, index=False)

print(f"Les différences ont été exportées vers {differences_output_path}")
