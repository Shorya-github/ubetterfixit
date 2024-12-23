from flask import Flask, render_template, request
import pandas as pd
import textdistance
import re
from collections import Counter

app = Flask(__name__)

# Step 1: Load and preprocess the data
with open('autocorrect book.txt', 'r', encoding='utf-8') as f:
    data = f.read().lower()
    words = re.findall(r'\w+', data)  # Use raw string here

# Step 2: Calculate word frequencies and probabilities
words_freq_dict = Counter(words)
total_count = sum(words_freq_dict.values())
probs = {word: freq / total_count for word, freq in words_freq_dict.items()}

@app.route('/')
def index():
    return render_template('index.html', suggestions=None)

@app.route('/suggest', methods=['POST'])
def suggest():
    keyword = request.form['keyword'].strip().lower()  # Handle empty or whitespace-only input
    if keyword:
        # Step 3: Calculate similarity scores for suggestions
        similarities = [
            1 - textdistance.Jaccard(qval=2).distance(word, keyword)
            for word in words_freq_dict.keys()
        ]

        # Step 4: Combine probabilities and similarities into a DataFrame
        df = pd.DataFrame({
            'Word': list(words_freq_dict.keys()),
            'Prob': list(probs.values()),
            'Similarity': similarities
        })

        # Step 5: Sort and select top suggestions
        suggestions = df.sort_values(['Similarity', 'Prob'], ascending=False).head(10)
        suggestions_list = suggestions[['Word', 'Similarity']].to_dict('records')  # Convert to list of dictionaries

        return render_template('index.html', suggestions=suggestions_list)

    return render_template('index.html', suggestions=None)  # Handle case where keyword is empty

if __name__ == '__main__':
    app.run(debug=True)
