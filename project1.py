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


print(data['language'].unique())

