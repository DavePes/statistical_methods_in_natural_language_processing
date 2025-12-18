from sacremoses import MosesTokenizer
from collections import Counter, defaultdict

files = ["guthenberg-cs.txt"]
langs = ["cs"]

## 
## Read files
texts = []
print("Reading files...")
for filename in files:
    # Open file and read content
    with open(filename, "r", encoding='latin-1') as f:
        texts.append(f.read())
## Tokenization with Moses
print("Tokenizing with Moses...")
def moses_tokenizer(texts):
  tokens = {}
  for i, lang in enumerate(langs):
    tok = MosesTokenizer(lang=lang)
    all_tokens = []
    # Iterate through sentences to tokenize them individually
    # for sentence in texts[i]:
    #     all_tokens.extend(tok.tokenize(sentence))
    all_tokens = tok.tokenize(texts[i])
    if lang == "cs":
        print(f"Original tokens in {lang}: {len(all_tokens)}")
        all_tokens = all_tokens[:15000] # Limit tokens
    if lang == "en":
        print(f"Original tokens in {lang}: {len(all_tokens)}")
        all_tokens = all_tokens[:30000] # Limit tokens
    tokens[lang] = all_tokens
    print(f"Tokenized {lang}: {len(all_tokens)} tokens")
  return tokens

tokens = moses_tokenizer(texts)

## Count unigrams, bigrams, trigrams
def dataset_counter(token_dict):
    # Initialize dictionaries to hold stats for all languages
    data_sizes = {}
    unigrams_frequencies, unigrams_counts = {}, {}
    bigrams_frequencies, bigrams_counts = {}, {}
    trigrams_frequencies, trigrams_counts = {}, {}

    for lang in langs:
        current_tokens = token_dict[lang]
        bigram = Counter()
        trigram = Counter()

        # Sliding window loop: Iterate through tokens to find N-grams
        for i in range(len(current_tokens)-1):
            # Create bigram tuple (w1, w2) and count it
            bigram[(current_tokens[i], current_tokens[i+1])] += 1

            # Check if we have enough tokens left for a trigram (w1, w2, w3)
            if i < len(current_tokens)-2:
                trigram[(current_tokens[i], current_tokens[i+1], current_tokens[i+2])] += 1

        # Count frequencies of individual words (Unigrams)
        unigrams_frequencies[lang] = Counter(current_tokens)
        unigrams_counts[lang] = len(unigrams_frequencies[lang]) # Count of unique words

        # Store Bigram stats
        bigrams_frequencies[lang] = bigram
        bigrams_counts[lang] = len(bigrams_frequencies[lang]) # Count of unique bigrams

        # Store Trigram stats
        trigrams_frequencies[lang] = trigram
        trigrams_counts[lang] = len(trigrams_frequencies[lang]) # Count of unique trigrams

        # Total number of tokens in the dataset
        data_sizes[lang] = len(current_tokens)

    return (unigrams_frequencies, unigrams_counts,
            bigrams_frequencies, bigrams_counts,
            trigrams_frequencies, trigrams_counts,
            data_sizes)

# # Run counter on Moses tokens
import math

# ... (Keep your imports, tokenization, and dataset_counter code as is) ...

# 1. Run Counter
(unigrams_frequencies, unigrams_counts,
 bigrams_frequencies, bigrams_counts,
 trigrams_frequencies, trigrams_counts,
 data_sizes) = dataset_counter(tokens)

# 2. Filter Unigrams (Apply Thresholds: cs=20, en=50)
thresholds = {"cs": 20, "en": 50}   
filtered_unigrams = {}
for lang in langs:
    filtered_unigrams[lang] = {}
    for word, freq in unigrams_frequencies[lang].items():
        if freq >= thresholds[lang]:
            filtered_unigrams[lang][word] = freq
print("Filtered unigrams counts:", {lang: len(filtered_unigrams[lang]) for lang in langs})


# embeddings
import fasttext
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

print("Loading models...")
# Suppress warnings for cleaner output
fasttext.FastText.eprint = lambda x: None
ft_models = {
    #'en': fasttext.load_model('cc.en.50.bin'),
    'cs': fasttext.load_model('cc.cs.50.bin')
}
# Visualization & Re-clustering Function ---
def pca_visualization(lang, classes_dict, ft_model):
    words = []
    brown_class_indices = [] # For coloring by Brown class
    vectors = []

    # Flatten the dictionary: Get list of words and their corresponding class index
    # classes_dict is { 'class_signature': ['word1', 'word2'], ... }
    for word in classes_dict:
        words.append(word)
        vectors.append(ft_model.get_word_vector(word))

    X = np.array(vectors)

    # --- PCA Projection ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

   

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    # Helper to plot scatter and labels
    def plot_scatter(ax, labels, title, cmap):
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, s=100, edgecolors='k', alpha=0.8)
        ax.set_title(title, fontsize=16)
        # Annotate words
        for i, word in enumerate(words):
            ax.text(X_pca[i, 0]+0.02, X_pca[i, 1]+0.02, word, fontsize=9, alpha=0.8)
        return scatter

    # Plot 1: Colors based on Brown Clustering (Original)
    plot_scatter(axes[0], brown_class_indices, f'Pca ({lang})', 'tab20')

    # Plot 2: Colors based on K-Means (Re-clustered)
    plot_scatter(axes[1], kmeans_labels, f'K-Means Re-clustering on Embeddings ({lang})', 'tab20')

    plt.tight_layout()
    plt.show()
def kmeans_clustering(X, n_clusters):
     # --- K-Means Re-clustering ---
    # We want 15 clusters as per assignment instructions
    n_clusters = 15
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    return kmeans.labels_
# Generate plots for each language
for lang in langs:
    print(f"\nGenerating plots for: {lang.upper()}")

    # Retrieve the clustering results we stored earlier
    #classes_dict = clustering_results[lang]

    # Pass the CLASSES, not the unigrams
    pca_visualization(lang, filtered_unigrams[lang], ft_models[lang])