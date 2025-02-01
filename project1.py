import pandas as pd
import numpy as np
import os
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA

# Ensure required packages are installed
try:
    from wordcloud import WordCloud
except ImportError:
    subprocess.check_call(["pip", "install", "wordcloud"])
    from wordcloud import WordCloud

# Ask for user choice
print("Please choose and only write the number: \n 1. Subject Wordmap \n 2. Emotion Analysis \n 3. Subject Frequency Bar Chart \n 4. Topic Clustering (LDA)")
users_choice = input().strip().lower()

# Load CSV file
file_path = 'SPGC-metadata-2018-07-18.csv'
metadata = pd.read_csv(file_path)

# Clean missing values in language and subjects columns
columns_to_update = [5, 7]  # Columns (index 5 & 7)
missing_values = {'', 'Missing', 'Unknown', 'set()'}
for col in columns_to_update:
    metadata.iloc[:, col] = metadata.iloc[:, col].replace(missing_values, 'Unknown').fillna('Unknown')

# Extract and clean subjects
subjects_list = metadata.iloc[:, 7].astype(str).tolist()
cleaned_subjects = []

for entry in subjects_list:
    cleaned_entry = entry.strip("{}").strip().lower()
    subjects = cleaned_entry.split('--') if '--' in cleaned_entry else cleaned_entry.split(',')
    subjects = [subject.strip().strip("'") for subject in subjects if subject.strip()]
    cleaned_subjects.extend(subjects)

# Remove unwanted values like "set()"
cleaned_subjects = [subject for subject in cleaned_subjects if subject and subject != "set()"]

# Create subject frequency dictionary
subject_counts = defaultdict(int)
for subject in cleaned_subjects:
    subject_counts[subject] += 1

# Convert to DataFrame for visualization
subject_df = pd.DataFrame(subject_counts.items(), columns=["Subject", "Frequency"])
subject_df = subject_df.sort_values(by="Frequency", ascending=False).head(20)  # Top 20 subjects

# ✅ Handle WordMap (Option 1)
if users_choice in ["1", "wordmap", "subject wordmap"]:
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(' '.join(cleaned_subjects))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Map of Subjects", fontsize=16)
    plt.savefig("wordcloud.png")
    plt.show()

# ✅ Handle Subject Frequency Bar Chart (Option 3)
elif users_choice in ["3", "bar chart", "subject frequency bar chart"]:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=subject_df["Frequency"], y=subject_df["Subject"], palette="viridis")
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Subjects", fontsize=12)
    plt.title("Top 20 Most Frequent Subjects", fontsize=14)
    plt.show()

# ✅ Handle Topic Clustering with LDA (Option 4)
elif users_choice in ["4", "lda", "topic clustering"]:
    # Convert subjects into a format suitable for LDA
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(cleaned_subjects)

    # Apply LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    topic_matrix = lda.fit_transform(X)

    # Assign each book to the most probable topic
    metadata["dominant_topic"] = topic_matrix.argmax(axis=1)

    # Count books per topic
    topic_counts = metadata["dominant_topic"].value_counts().sort_index()
    print("\n📌 Number of books per topic:")
    print(topic_counts)

    # Display top words for each topic
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        print(f"\nTopic {topic_idx + 1}: ", [words[i] for i in topic.argsort()[-10:]])

    # ✅ Generate WordClouds for each topic
    plt.figure(figsize=(10, 5))
    for i, topic in enumerate(lda.components_):
        topic_words = {words[j]: topic[j] for j in topic.argsort()[-15:]}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
        plt.subplot(1, 5, i + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {i+1}")

    plt.show()

    # ✅ Plot the topic distribution
    plt.figure(figsize=(10, 5))
    sns.barplot(x=topic_counts.index, y=topic_counts.values, palette="viridis")
    plt.xlabel("LDA Topic")
    plt.ylabel("Number of Books")
    plt.title("Distribution of Books Across LDA Topics")
    plt.show()

    # ✅ Reduce topic matrix to 2D using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(topic_matrix)

    # Create a DataFrame for visualization
    pca_df = pd.DataFrame({
        "x": X_pca[:, 0],
        "y": X_pca[:, 1],
        "Topic": metadata["dominant_topic"],
        "Book Title": metadata["title"].fillna("Unknown")
    })

    # ✅ Generate an interactive scatter plot
    fig = px.scatter(pca_df, x="x", y="y", color=pca_df["Topic"].astype(str),
                     hover_data={"Book Title": True, "x": False, "y": False},
                     title="Interactive PCA Clustering of Book Topics",
                     labels={"Topic": "LDA Topic"})
    fig.show()

else:
    print("❌ Invalid choice. Please enter '1' for Wordmap, '3' for Bar Chart, or '4' for LDA Topic Clustering.")



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

