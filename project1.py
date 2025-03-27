print ("Welcome. Please choose and only write the number: \n 1. Subject Wordmap \n 2. emotion analysis \n 3. Subject Frequency Bar Chart \n 4. Topic Clustering (LDA)")
users_choice = input ()


import subprocess
import pandas as pd
try: 
    from wordcloud import WordCloud
except ImportError:
    subprocess.check_call(["pip", "install", "wordcloud"])
    from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Seaborn library is specifically designed for creating attractive and informative statistical graphics.
import seaborn as sns
# counter dictionary subclass is used for counting hashable objects (e.g., strings, numbers).
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# os is used for handling the path in the emotion analysis section.
import os
import csv
try:
    import spacy
    from spacy.cli import download
except ImportError:
    subprocess.check_call(["pip", "install", "spacy"])
    import spacy
    from spacy.cli import download

# Ensure the 'en_core_web_lg' model is installed and load it
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    download("en_core_web_lg")  
    nlp = spacy.load("en_core_web_lg")

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

# Convert to a format suitable for LDA
subject_corpus = [" ".join(subject.split()) for subject in cleaned_subjects]

# Convert to DataFrame for visualization
subject_df = pd.DataFrame(subject_counts.items(), columns=["Subject", "Frequency"])
subject_df = subject_df.sort_values(by="Frequency", ascending=False).head(20)  # Top 20 subjects

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
    print ('wordcloud file is ready!')




if users_choice in ["3", "3.", "subject frequency bar chart", "bar chart"]:
    plt.figure(figsize=(16, 12))  # Wider than tall
    ax = sns.barplot(x="Frequency", y="Subject", 
                    data=subject_df.sort_values("Frequency", ascending=False),
                    palette="viridis")
    plt.title("Top 20 Subjects by Frequency", fontsize=14)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("")  # Remove redundant y-label
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    plt.savefig("subject_frequencies.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.savefig("barplot.png")





if users_choice in ["3", "bar chart", "subject frequency bar chart"]:
    plt.figure(figsize=(12, 18))
    sns.barplot(x=subject_df["Frequency"], y=subject_df["Subject"], palette="viridis")
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Subjects", fontsize=12)
    plt.title("Top 20 Most Frequent Subjects", fontsize=14)
    plt.show()

if users_choice in ["4", "Topic Clustering", "four"]:
    lda = None
    #  Topic Clustering with LDA (Choice 4)
    if users_choice in ["4", "lda", "topic clustering"]:
        # Convert subjects into a format suitable for LDA
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(subject_corpus)

        # Apply LDA
        num_topics = 5 # The number is changeable & I should read more, like elbow method, or based on Perplexity Score, etc.
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        topic_matrix = lda.fit_transform(X) #trains model & transforms the data

        # Get feature names
        words = vectorizer.get_feature_names_out()
        
        # Display top words for each topic
        words = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            print(f"\nTopic {topic_idx + 1}: ", [words[i] for i in topic.argsort()[-10:]])

        # WordCloud for LDA Topics
        plt.figure(figsize=(12, 6))
        for i, topic in enumerate(lda.components_):
            topic_words = {words[j]: topic[j] for j in topic.argsort()[-15:]}
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
            
            plt.subplot(1, num_topics, i+1)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"Topic {i+1}")
        plt.tight_layout()  # Fix layout issues
        plt.show()

    #     # using preplexity score to find the best number
    #     perplexities = []
    #     topic_range = range(2, 10)  # Testing from 2 to 9 topics

    #     for num in topic_range:
    #         lda = LatentDirichletAllocation(n_components=num, random_state=42)
    #         lda.fit(X)
    #         perplexities.append(lda.perplexity(X))

    #     # Plot Perplexity Score
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(topic_range, perplexities, marker='o', linestyle='--')
    #     plt.xlabel('Number of Topics')
    #     plt.ylabel('Perplexity Score')
    #     plt.title('Finding the Best Number of Topics')
    # plt.savefig("perplexity_score.png")
    # plt.show()
    # print("Perplexity Score plot saved as 'perplexity_score.png'")

        # Improving meaningfulness of clusters by more descriptive visualization of them
        # Labeling each cluster based on its themes
    topic_labels = [
        "19th-Century Science and Drama: English and French Literary Fiction",
        "Historical and Juvenile Fiction Across [American] Nations",
        "Cultural & Social Criticism [American Context]",
        "Classic British Literature & Biographies",
        "Christian & War-Time Fiction [European Context]"
    ]

    for topic_idx, topic in enumerate(lda.components_):
        print(f"\n📖 Topic {topic_idx + 1} - {topic_labels[topic_idx]}:")
        print(" 🔹 ", [words[i] for i in topic.argsort()[-10:]])

        # Create DataFrame of topic-word weights
        topic_word_matrix = pd.DataFrame(lda.components_, index=topic_labels, columns=words)

        # Plot heatmap
        plt.figure(figsize=(30, 18))
        sns.heatmap(topic_word_matrix.iloc[:, :20], cmap="coolwarm", annot=False, xticklabels=True)
        plt.xlabel("Words")
        plt.ylabel("Topics")
        plt.yticks(rotation=45)
        plt.tick_params(axis='y', labelsize=10)  # Reduce font size for y-axis labels
        plt.title("Topic-Word Distribution")
        plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig("LDA_heatmap.png")  # ✅ Save plot BEFORE plt.show()
    plt.show()
    print("✅ LDA heatmap saved as 'LDA_heatmap.png'")



        # Assign each book to a topic
    book_topic_matrix = pd.DataFrame(topic_matrix, columns=topic_labels)
    metadata["Dominant Topic"] = book_topic_matrix.idxmax(axis=1)

        # Show a sample of results
    print("Books with Assigned Topics:")
    print(metadata[["title", "Dominant Topic"]].head(20))

    # Save Word Clouds for Each Topic
    plt.figure(figsize=(12, 6))
    for i, topic in enumerate(lda.components_):
        topic_words = {words[j]: topic[j] for j in topic.argsort()[-15:]}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
        
        plt.subplot(1, num_topics, i+1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {i+1}")

    plt.tight_layout()
    plt.savefig("LDA_wordclouds.png") 
    plt.show()
    print("✅ Word clouds saved as 'LDA_wordclouds.png'")

 
    print("Current working directory:", os.getcwd())


if users_choice in ["2", "2.", "emotion analysis"]:
    def lemmatize_word(word):
        doc = nlp(word)
        return doc[0].lemma_ if doc else word

    def perform_text_analysis():
        print("Please write the ID (e.g., 10 or 14). Please choose a number that is available in the folder of Counts.")
        file_id = input().strip()
        
        text_folder_path = "SPGC-counts-2018-07-18"
        file_name = f"PG{file_id}_counts.txt"
        full_path = os.path.join(text_folder_path, file_name)

        try:
            df = pd.read_csv(full_path, sep="\t", names=["word", "count"], 
                           engine="python", header=None, skiprows=1)
            df["lemma"] = df["word"].apply(lemmatize_word)
            df_final = df[["lemma", "count"]]
            
            output_file = f"PG{file_id}_counts_lemmatized.txt"
            output_path = os.path.join(text_folder_path, output_file)
            df_final.to_csv(output_path, sep="\t", index=False, header=False)
            
            print(f"Lemmatized file saved as {output_file}")
            return file_id, output_path  # Return both values

        except FileNotFoundError:
            print(f"File '{full_path}' not found.")
            return None, None

    # Get the values from the function
    file_id, full_path = perform_text_analysis()
    
    if file_id:  
        text_folder_path = "SPGC-counts-2018-07-18"
        file_name = f"PG{file_id}_counts_lemmatized.txt"  
        full_path = os.path.join(text_folder_path, file_name)
        
       
        

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

    text_folder_path = r"SPGC-counts-2018-07-18"
    file_name = f"PG{file_id}_counts_lemmatized.txt" # Use the lemmatized file name
    full_path = os.path.join(text_folder_path, file_name)  # Combine folder and file name   
    text_result = analysis (full_path)
        # print (text_result)

    

        # Create a dynamic filename using the selected ID
    output_filename = f"PG{file_id}_results.tsv"
    with open (output_filename,'w', newline='', encoding='utf-8') as tsv_file: 
            writer = csv.writer (tsv_file, delimiter= '\t')
            writer.writerow(['Word', 'Count', 'Emotions'])
            for word, values in text_result.items():
                parts = values.split(", ", 1)  # Splitting only on the first comma
                 # Assign values correctly
                count = parts[0]  # First part is the count
                emotions = parts[1] if len(parts) > 1 else ""  # Second part is emotions, if available

                 # Remove extra brackets from emotions
                emotions = emotions.strip("[]").replace("'", "")  # Cleaning up emotion formatting
    
                  # Write to file
                writer.writerow([word, count, emotions])

    print(f"Emotion analysis saved.")          


    def emotion_frequency (output_path):
            emotions_in_text = {}
            with open (output_path) as tsv_file:
                reader = csv.reader (tsv_file, delimiter='\t')
                next (reader)   # Skip the header row
                for row in reader:
                    word, number, emotions = row
                    separated_emotions= emotions.strip().split(',')
                    for emotion in separated_emotions:
                        if emotion in ['anticipation', 'joy', 'positive', 'surprise', 'trust', 'anger', 'negative', 'disgust', 'fear', 'sadness']:
                          # Increment the count for the emotion
                            if emotion in emotions_in_text:
                                   emotions_in_text[emotion] += int(number)
                            else:
                                   emotions_in_text[emotion] = int(number)
            return emotions_in_text
        
    emotion_frequency =  emotion_frequency (output_filename) 
    print (emotion_frequency)


    emotions = list(emotion_frequency.keys())
    values = list(emotion_frequency.values())
    sns.set(style="whitegrid")
        # Create a bar chart
    plt.figure(figsize=(10, 6))  # Set the figure size
    bars = plt.bar(emotions, values, color=sns.color_palette("Blues", n_colors=len(emotions)))

        # Adding titles and labels
    plt.title('Emotion Frequency Distribution', fontsize=18, weight='bold', family='serif')
    plt.xlabel('Emotions', fontsize=12, weight='bold', family='serif')
    plt.ylabel('Frequency', fontsize=12, weight='bold', family='serif')

        # Rotate the x-axis labels for better visibility
    plt.xticks(rotation=45, fontsize=12)

        # Adding grid lines to make the chart more readable
    plt.grid(axis='y', linestyle='--', alpha=0.7) 

    for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 100, round(yval, 0), ha='center', va='bottom', fontsize=10)

        # Display the chart
    plt.tight_layout()
        # plt.show()        
    plt.savefig(f"barchart{file_id}.png")  
    print ('The barchart is saved.')

                        


   
