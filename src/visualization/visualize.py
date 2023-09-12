from matplotlib import pyplot as plt
import os
import sys
# Add the directory containing the package to the system path
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(package_dir)

from data.make_dataset import freq_df, wordcloud, freq_df_updated

figure_path = "reports/figures"


## Creating a Frequency Diagram with matplotlib
# Assuming you have already defined freq_df and highlighted_words
highlighted_words = ["not", "seats", "seat"]

plt.figure(figsize=(10, 6))
colors = ['lightblue' if token in highlighted_words else 'gray' for token in freq_df.index]
plt.barh(freq_df.head(15).index, freq_df.head(15)["freq"], color=colors)
plt.xlabel("Frequency")
plt.ylabel("Token")
plt.title("Top 10 Most Common Words")
plt.gca().invert_yaxis()

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the "reports/figures" directory
relative_path = os.path.join(script_directory, "../../reports/figures")

# Make sure the directory exists, and if not, create it
os.makedirs(relative_path, exist_ok=True)

# Save the figure in the "reports/figures" directory
plt.savefig(os.path.join(relative_path, "frequency_diagram.png"))


wordcloud(freq_df["freq"], max_words=50, save_path=figure_path, figure_name="wordcloud_1.png")
# treats the 50 most frequent words of the complete corpus as stop words
wordcloud(freq_df["freq"], max_words=50, stopwords=freq_df.head(50).index, save_path=figure_path, figure_name="wordcloud_2.png") 

# wordcloud(freq_df["tfidf"], max_words=100)
# # treats the 50 most frequent words of the complete corpus as stop words
# wordcloud(freq_df["tfidf"], max_words=100, stopwords=freq_df.head(50).index) 

wordcloud(freq_df_updated['tfidf'], max_words=50, save_path=figure_path, figure_name="wordcloud_3.png")
# treats the 50 most frequent words of the complete corpus as stop words
wordcloud(freq_df_updated["tfidf"], max_words=50, stopwords=freq_df.head(50).index, save_path=figure_path, figure_name="wordcloud_4.png") 