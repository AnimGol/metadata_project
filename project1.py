import pandas as pd

import matplotlib.pyplot as plt

# Seaborn library is specifically designed for creating attractive and informative statistical graphics.
import seaborn as sns

# counter dictionary subclass is used for counting hashable objects (e.g., strings, numbers).
from collections import Counter

# 'r' tells Python to treat the backslashes as literal characters.
file_path = r'SPGC-metadata-2018-07-18.csv'
metadata = pd.read_csv(file_path)

# print(metadata.head())

# Extract the sixth column which is related to languages (index 5) from the metadata 
language_list = metadata.iloc[:, 5].tolist()

#print(language_list)


# defining subject as list.
subjects_list = metadata.iloc[:, 7].tolist()

#print(subjects_list)


#finfing inconsistency in language row
# print(metadata['language'].unique())

columns_to_update = [5, 7]  # Columns 6 and 8 (zero-indexed)
missing_values = {'', 'Missing', 'Unknown', 'set()'}
# Replace missing data in specified columns with "Unknown"
for col in columns_to_update:
    metadata.iloc[:, col] = metadata.iloc[:, col].replace(missing_values, 'Unknown')
    metadata.iloc[:, col] = metadata.iloc[:, col].fillna('Unknown')  # Replace NaN with "Unknown"
output_file_path =  r'cleaned_metadata.csv'
metadata.to_csv(output_file_path, index=False)
print(f"Cleaned metadata saved to: {output_file_path}")


