# author: Michael Hüppe
# date: 22.11.2024
# project: dataAnalysis/extractTopics.py
import json

import pandas as pd

from keybert import KeyBERT

def extract_keywords_with_keybert(kw_model, abstract, n=5):
    # Initialize KeyBERT model
    # Extract keywords
    keywords = kw_model.extract_keywords(
        abstract,
        stop_words="english",          # Remove common stop words
        use_maxsum=False,              # Maximal Marginal Relevance (MMR) for diversity (optional)
        use_mmr=True,                  # Use MMR to diversify results
        diversity=0.7,                 # Controls diversity in MMR
        top_n=n                        # Number of keywords to extract
    )
    return keywords

if __name__ == '__main__':
    dataPath = r"data/ML-Arxiv-Papers.csv"
    df = pd.read_csv(dataPath)

    kw_model = KeyBERT("all-MiniLM-L6-v2")  # Use a smaller Sentence-BERT model for efficiency
    topics = []
    titles = df["title"]
    abstracts = df["abstract"]
    import time

    # Number of iterations
    total_iterations = len(df)
    # Start the loop
    start_time = time.time()
    papers = {}

    for i in range(1, total_iterations + 1):
        idx = i-1
        papers[titles[idx]] = {
            "topics": extract_keywords_with_keybert(kw_model, abstracts[idx], n=15),
            "abstract": abstracts[idx]
        }
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Calculate remaining time
        avg_time_per_iteration = elapsed_time / i
        remaining_time = avg_time_per_iteration * (total_iterations - i)

        # Print progress and ETA
        print(f"Iteration {i}/{total_iterations} - ETA: {remaining_time:.2f} seconds")
    json.dump(papers, open(dataPath.replace(".csv", ".json"), "w"))