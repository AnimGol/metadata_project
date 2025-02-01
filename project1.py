print ("Please choose and only write the number: \n 1. Subject Wordmap \n 2. emotion analysis \n 3. Subject Frequency Bar Chart")
users_choice = input ()



import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Seaborn library is specifically designed for creating attractive and informative statistical graphics.
import seaborn as sns
# counter dictionary subclass is used for counting hashable objects (e.g., strings, numbers).
from collections import defaultdict
# os is used for handling the path in the emotion analysis section.
import os

# 'r' tells Python to treat the backslashes as literal characters.
file_path = r'SPGC-metadata-2018-07-18.csv'
metadata = pd.read_csv(file_path)

# print(metadata.head())




#finding inconsistency in language row
# print(metadata['language'].unique())

# cleaning data and handling missing info
columns_to_update = [5, 7]  # Columns 6 and 8 (zero-indexed)
missing_values = {'', 'Missing', 'Unknown', 'set()'}
# Replace missing data in specified columns with "Unknown"
for col in columns_to_update:
    metadata.iloc[:, col] = metadata.iloc[:, col].replace(missing_values, 'Unknown')
    metadata.iloc[:, col] = metadata.iloc[:, col].fillna('Unknown')  # Replace NaN with "Unknown"
output_file_path =  r'cleaned_metadata.csv'
metadata.to_csv(output_file_path, index=False)
# print(f"Cleaned metadata saved to: {output_file_path}")

# Extract the sixth column which is related to languages (index 5) from the metadata 
language_list = metadata.iloc[:, 5].tolist()
#print(language_list)





# Defining subjects as a list
subjects_list = metadata.iloc[:, 7].astype(str).tolist()  # Convert to string to avoid NaN issues


# Cleaning and spliting subjects
cleaned_subjects = []
for entry in subjects_list:
    # Remove curly braces `{}` and normalize text
    cleaned_entry = entry.strip("{}").strip().lower()

    # Split by both ',' and '--' to handle different formats
    if '--' in cleaned_entry:
        subjects = cleaned_entry.split('--')
    else:
        subjects = cleaned_entry.split(',')

    # Clean extra spaces and remove empty values
    subjects = [subject.strip().strip("'") for subject in subjects if subject.strip()]
    
    # Add cleaned subjects to the list
    cleaned_subjects.extend(subjects)


# Remove unwanted values like "set()"
cleaned_subjects = [subject for subject in cleaned_subjects if subject and subject != "set()"]

# Join all subjects into a single string for Word Cloud
subject_text = ' '.join(cleaned_subjects)

# Create a subject frequency dictionary
subject_counts = defaultdict(int)
for subject in cleaned_subjects:
    subject_counts[subject] += 1

# Convert to DataFrame for visualization
subject_df = pd.DataFrame(subject_counts.items(), columns=["Subject", "Frequency"])
subject_df = subject_df.sort_values(by="Frequency", ascending=False).head(20)  # Top 20 subjects

# User choice handling
if users_choice in ["1", "1.", "subject wordmap", "wordmap"]:
    # Create and display a word cloud
    subject_text = ' '.join(cleaned_subjects)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(subject_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Map of Subjects", fontsize=16)
    plt.savefig("wordcloud.png")
    plt.show()


if users_choice == "1" or users_choice == "1." or users_choice == "1. Subject Wordmap" or users_choice == "one" or users_choice == "Subject Wordmap":
    # Create the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(subject_text)
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    # "bilinear" smooths the edges and makes the word cloud look better when stretched to fit the figure. Without it, the image might look pixelated or blocky
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide the axes
    plt.title("Word Map of Subjects", fontsize=16)
    plt.savefig("wordcloud.png")
    plt.show()


if users_choice in ["3", "3.", "subject frequency bar chart", "bar chart"]:
    # Create and display a bar chart
        # Create and display a bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=subject_df["Frequency"], y=subject_df["Subject"], palette="viridis", legend=False)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Subjects", fontsize=12)
    plt.title("Top 20 Most Frequent Subjects", fontsize=14)
    plt.show()
    plt.savefig("barplot.png")

else:
    print("Invalid choice. Please enter '1' for Wordmap or '2' for Bar Chart.")




if users_choice in ["2", "2.", "2. emotion analysis", "two", "emotion analysis"]:
    print ("Please write the ID (e.g., 10 or 14). Please choose a number that is availavle in the folder of Counts.")
    id = input (). strip ()
    text_folder_path = r"SPGC-counts-2018-07-18"
    file_name = f"PG{id}_counts.txt"
    full_path = os.path.join(text_folder_path, file_name)  # Combine folder and file name
    try:
        print(f"Trying to open file: {full_path}")
        # Open and read the file
        with open(full_path, "r") as file:
            content = file.read()
            print ('file is found and read successfully!')

        def emotion_dictionary(file_path):
                emotion_lexicon = {}
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        for line in file:
                            word, emotion, value = line.strip().split('\t')
                            if value == '1':
                                if word not in emotion_lexicon:
                                    emotion_lexicon[word] = []
                                emotion_lexicon[word].append(emotion)
                except FileNotFoundError:
                    print(f"Emotion lexicon file '{file_path}' not found. Please check the path.")
                return emotion_lexicon
        result = emotion_dictionary(r'Lexicon.txt')
        # print (result)

        def analysis(full_path):
            second_lexicon = {}
            with open(full_path, 'r', encoding='utf-8') as file:
                for line in file:
                    word, count = line.strip().split('\t')
                    emotions = result.get(word, [])  # Get emotions if the word exists in the emotion lexicon
                    second_lexicon[word] = f"{count}, {emotions}"  # Store count and emotions as a string
            return second_lexicon

        
        text_result = analysis (full_path)
        print (text_result)
                        


    except FileNotFoundError:
        print(f"File '{full_path}' not found. Please check the ID and try again.")

