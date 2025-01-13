import pandas as pd
import matplotlib.pyplot as plt
# Seaborn library is specifically designed for creating attractive and informative statistical graphics.
import seaborn as sns
# counter dictionary subclass is used for counting hashable objects (e.g., strings, numbers).
from collections import Counter

# 'r' tells Python to treat the backslashes as literal characters.
file_path = r'C:\Users\minag\OneDrive\Desktop\metadata_project\SPGC-metadata-2018-07-18.csv'
metadata = pd.read_csv(file_path)