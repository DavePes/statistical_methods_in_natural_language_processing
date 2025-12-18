
#!pip install sacremoses
from sacremoses import MosesTokenizer
from collections import Counter, defaultdict



# !wget -O "guthenberg-cs.txt" "https://www.gutenberg.org/cache/epub/34225/pg34225.txt"
# !wget -O "guthenberg-en.txt" "https://www.gutenberg.org/cache/epub/37536/pg37536.txt"
files = ["guthenberg-cs.txt", "guthenberg-en.txt"]
langs = ["cs", "en"]

## 
## Read files
texts = []
print("Reading files...")
for filename in files:
    # Open file and read content
    with open(filename, 'r', encoding='utf-8') as f:
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

import math
from collections import defaultdict

import math
from collections import defaultdict, Counter

class BrownClustering:
    def __init__(self, unigram_freqs, bigram_freqs, total_tokens):
        self.total_tokens = total_tokens
        # STORE UNIGRAMS FOR SORTING LATER
        self.unigram_freqs = unigram_freqs
        
        # 1. FLATTEN BIGRAMS
        self.bigram_counts_flat = {}
        first_val = next(iter(bigram_freqs.values())) if bigram_freqs else 0
        
        if isinstance(first_val, (dict, Counter)):
            for w1, inner in bigram_freqs.items():
                for w2, count in inner.items():
                    if count > 0: self.bigram_counts_flat[(w1, w2)] = count
        else:
            self.bigram_counts_flat = {k: v for k, v in bigram_freqs.items() if v > 0}

        # 2. INITIALIZE MAPPINGS
        # We map ALL words to IDs, but we don't activate them yet.
        self.word_to_id = {w: i for i, w in enumerate(unigram_freqs.keys())}
        self.id_to_words = {i: [w] for w, i in self.word_to_id.items()}
        
        # CHANGED: Start empty. We will populate this in run_greedy using the "Trick 4" window.
        self.active_ids = set()
        
        # C1: Unigram counts {class_id: count}
        self.c1 = {self.word_to_id[w]: c for w, c in unigram_freqs.items()}
        
        # C2 & C2_REV: Full Graph (Static counts for all words)
        self.c2 = defaultdict(lambda: defaultdict(int))
        self.c2_rev = defaultdict(lambda: defaultdict(int))
        
        for (w1, w2), count in self.bigram_counts_flat.items():
            if w1 in self.word_to_id and w2 in self.word_to_id:
                u, v = self.word_to_id[w1], self.word_to_id[w2]
                self.c2[u][v] = count
                self.c2_rev[v][u] = count

        # Q, S, L: Initialized empty
        self.q = defaultdict(lambda: defaultdict(float))
        self.s = defaultdict(float)
        self.L = defaultdict(dict)

    def get_q(self, u, v):
        """Calculates q(u, v) = p(u,v) * log(p(u,v) / (p(u)p(v)))."""
        count = self.c2.get(u, {}).get(v, 0)
        if count == 0: return 0.0
        
        p_uv = count / self.total_tokens
        p_u = self.c1[u] / self.total_tokens
        p_v = self.c1[v] / self.total_tokens
        
        return p_uv * math.log2(p_uv / (p_u * p_v))

    def compute_current_global_mi(self):
        """Calculates sum of all q(u,v) for current active classes."""
        total_mi = 0.0
        for u in self.active_ids:
            for v in self.c2[u]:
                total_mi += self.get_q(u, v)
        return total_mi

    def initialize_structures(self):
        """Initial calculation of Q, S, and L matrices for CURRENT active_ids."""
        print(f"Initializing Q, S, L for {len(self.active_ids)} classes...")
        
        # 1. Compute Q and S
        for u in self.active_ids:
            for v in self.c2[u]:
                # Only compute interaction if v is also active or we are tracking full graph
                # Optimization: In Brown, we only care about Q(active, active).
                if v in self.active_ids: 
                    self.q[u][v] = self.get_q(u, v)
        
        for u in self.active_ids:
            sum_row = sum(self.q[u].values())
            sum_col = sum(self.q[k][u] for k in self.c2_rev[u] if k in self.active_ids) 
            self.s[u] = sum_row + sum_col - self.q[u].get(u, 0.0)

        # 2. Compute L for all pairs
        ids = list(self.active_ids)
        # Clear L to be safe
        self.L = defaultdict(dict)
        
        total_pairs = len(ids) * (len(ids) - 1) // 2
        
        for i in range(len(ids)):
            if i % 100 == 0: print(f"Init L: row {i}/{len(ids)}", end='\r')
            for j in range(i + 1, len(ids)):
                self.update_L_entry(ids[i], ids[j])
        print("\nInitialization complete.")

    def update_L_entry(self, u, v):
        """Computes L[u][v] = sub(u,v) - add(u,v). Fully Optimized."""
        sub = self.s[u] + self.s[v] - self.q[u].get(v, 0.0) - self.q[v].get(u, 0.0)
        
        c1_new = self.c1[u] + self.c1[v]
        p_new = c1_new / self.total_tokens
        inv_total = 1.0 / self.total_tokens
        add = 0.0
        
        outgoing = set(self.c2[u].keys()) | set(self.c2[v].keys())
        incoming = set(self.c2_rev[u].keys()) | set(self.c2_rev[v].keys())
        
        neighbors = outgoing | incoming
        neighbors.add(u)
        neighbors.discard(v) 
        
        for k in neighbors:
            if k == u or k == v: continue 
            if k not in self.active_ids: continue

            p_k = self.c1[k] * inv_total

            c2_new_k = self.c2[u].get(k, 0) + self.c2[v].get(k, 0)
            if c2_new_k > 0:
                p_new_k = c2_new_k * inv_total
                add += p_new_k * math.log2(p_new_k / (p_new * p_k))
            
            c2_k_new = self.c2[k].get(u, 0) + self.c2[k].get(v, 0)
            if c2_k_new > 0:
                p_k_new = c2_k_new * inv_total
                add += p_k_new * math.log2(p_k_new / (p_k * p_new))

        c2_self = self.c2[u].get(u,0) + self.c2[u].get(v,0) + \
                  self.c2[v].get(u,0) + self.c2[v].get(v,0)
        
        if c2_self > 0:
            p_self = c2_self * inv_total
            add += p_self * math.log2(p_self / (p_new * p_new))
            
        self.L[u][v] = sub - add

    def run_greedy(self, target_classes, lang_name="Unknown"):
        f = open("brown_log.txt", "w", encoding="utf-8")
        # 1. Sort words by frequency (Trick 4 requirement)
        sorted_words = sorted(self.unigram_freqs, key=self.unigram_freqs.get, reverse=True)
        
        # 2. Split into Initial K and Remaining
        initial_words = sorted_words[:target_classes]
        remaining_words = sorted_words[target_classes:]
        
        print(f"[{lang_name}] V={len(sorted_words)}, K={target_classes}. Streaming {len(remaining_words)} words.")

        # 3. Initialize with Top K
        self.active_ids = {self.word_to_id[w] for w in initial_words}
        self.initialize_structures() # O(K^2)
        
        next_id = max(self.word_to_id.values()) + 1
        merge_count = 0
        
        # 4. Stream Remaining Words
        # Logic: Add 1 word (K+1) -> Merge 1 pair (K)
        
        words_to_process = remaining_words
        # We process remaining words, then stop. 
        # (Or optionally merge K -> 1 at the very end, but usually we return K classes)
        
        for step, w in enumerate(words_to_process):
            new_u = self.word_to_id[w]
            self.active_ids.add(new_u)
            
            # --- Incremental Update for new_u (Trick 3/4) ---
            # Calculate Q and S for the new single node against existing K
            s_new = 0.0
            
            for v in self.active_ids:
                if v == new_u: continue
                
                # Compute Q(new, v) and Q(v, new)
                q_uv = self.get_q(new_u, v)
                q_vu = self.get_q(v, new_u)
                
                if q_uv != 0: self.q[new_u][v] = q_uv
                if q_vu != 0: self.q[v][new_u] = q_vu
                
                s_new += q_uv + q_vu
                
                # Update S[v] (Row v gained a column new_u, Col v gained a row new_u)
                self.s[v] += (q_uv + q_vu)
                
            # Self loop
            q_self = self.get_q(new_u, new_u)
            self.q[new_u][new_u] = q_self
            s_new += q_self
            self.s[new_u] = s_new - q_self # s definition: row + col - self

            # Calculate L for new_u against all v
            # (We only need to update the new row in L, not the whole matrix)
            for v in self.active_ids:
                if v != new_u:
                    # Maintain triangular matrix key (min, max)
                    if v > new_u: self.update_L_entry(new_u, v)
                    else: self.update_L_entry(v, new_u)

            # --- Perform Merge (Reduce K+1 -> K) ---
            if len(self.active_ids) > target_classes:
                # Find pair with Min Loss
                best_pair = None
                min_loss = float('inf')
                
                for u in self.L:
                    for v, loss in self.L[u].items():
                        if loss < min_loss:
                            min_loss = loss
                            best_pair = (u, v)
                
                if best_pair:
                    u, v = best_pair
                    w_u = self.id_to_words[u] if len(self.id_to_words[u]) < 3 else f"Cluster_{u}"
                    w_v = self.id_to_words[v] if len(self.id_to_words[v]) < 3 else f"Cluster_{v}"
                    print(f"Step {step}/{len(remaining_words)} | Merge: {w_u} + {w_v} | Loss: {min_loss:.6f}")      
                    f.write(f"{merge_count}\t{w_u}\t{w_v}\t{min_loss:.6f}\n")
                    # MERGE LOGIC
                    new_id = next_id
                    next_id += 1
                    
                    self.id_to_words[new_id] = self.id_to_words[u] + self.id_to_words[v]
                    del self.id_to_words[u]
                    del self.id_to_words[v]
                    
                    self.c1[new_id] = self.c1[u] + self.c1[v]
                    
                    # Update C2 / C2_REV
                    row_neighbors = set(self.c2[u].keys()) | set(self.c2[v].keys())
                    for k in row_neighbors:
                        val = self.c2[u][k] + self.c2[v][k]
                        self.c2[new_id][k] = val
                        self.c2_rev[k][new_id] = val 
                    
                    col_neighbors = set(self.c2_rev[u].keys()) | set(self.c2_rev[v].keys())
                    for k in col_neighbors:
                        if k == u or k == v: continue
                        val = self.c2[k][u] + self.c2[k][v]
                        self.c2[k][new_id] = val
                        self.c2_rev[new_id][k] = val 

                    self_val = self.c2[u].get(u,0) + self.c2[u].get(v,0) + \
                               self.c2[v].get(u,0) + self.c2[v].get(v,0)
                    self.c2[new_id][new_id] = self_val
                    self.c2_rev[new_id][new_id] = self_val

                    # Update Q and S for NEW class
                    s_new_merge = 0.0
                    q_new_k = {}
                    q_k_new = {}
                    
                    for k in self.active_ids:
                        if k == u or k == v: continue
                        qnk = self.get_q(new_id, k)
                        qkn = self.get_q(k, new_id)
                        q_new_k[k] = qnk
                        q_k_new[k] = qkn
                        s_new_merge += qnk + qkn
                        
                        # Update S[k] - remove old u/v interactions, add new_id interaction
                        delta = (qkn + qnk) - (self.q[k].get(u,0) + self.q[k].get(v,0) + 
                                               self.q[u].get(k,0) + self.q[v].get(k,0))
                        self.s[k] += delta
                        
                    q_self = self.get_q(new_id, new_id)
                    s_new_merge += q_self 
                    self.s[new_id] = s_new_merge - q_self # s is row+col-self
                    
                    for k, val in q_new_k.items(): self.q[new_id][k] = val
                    for k, val in q_k_new.items(): self.q[k][new_id] = val
                    self.q[new_id][new_id] = q_self

                    # Cleanup
                    self.active_ids.remove(u)
                    self.active_ids.remove(v)
                    self.active_ids.add(new_id)
                    
                    if u in self.L: del self.L[u]
                    if v in self.L: del self.L[v]
                    for k in self.L:
                        if u in self.L[k]: del self.L[k][u]
                        if v in self.L[k]: del self.L[k][v]

                    # Update Loss for new row
                    for k in self.active_ids:
                        if k == new_id: continue
                        if k > new_id: self.update_L_entry(new_id, k)
                        else: self.update_L_entry(k, new_id)

        print(f"\n[{lang_name}] Clustering complete. Final classes: {len(self.active_ids)}")
        return self.id_to_words

## 
# 2. Run Brown Clustering for each language
clustering_results = {}
for lang in langs:
    print(f"\nProcessing {lang}...")
    
    u_freqs = unigrams_frequencies[lang]
    b_freqs = bigrams_frequencies[lang]
    n_tokens = data_sizes[lang]
    
    algo = BrownClustering(u_freqs, b_freqs, n_tokens)
    clustering_results[lang] = algo.run_greedy(15, lang)

# embeddings
# !pip install fasttext
import fasttext
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Download and Load FastText Models ---
print("Downloading FastText models (50-dim)...")
# !wget -nc "https://ufallab.ms.mff.cuni.cz/~helcl/npfl147/cc.en.50.bin"
# !wget -nc "https://ufallab.ms.mff.cuni.cz/~helcl/npfl147/cc.cs.50.bin"

print("Loading models...")
# Suppress warnings for cleaner output
fasttext.FastText.eprint = lambda x: None 
ft_models = {
    'en': fasttext.load_model('cc.en.50.bin'),
    'cs': fasttext.load_model('cc.cs.50.bin')
}

# Visualization & Re-clustering Function ---
def process_and_visualize(lang, classes_dict, ft_model):
    words = []
    brown_class_indices = [] # For coloring by Brown class
    vectors = []
    
    # Flatten the dictionary: Get list of words and their corresponding class index
    # classes_dict is { 'class_signature': ['word1', 'word2'], ... }
    for class_idx, (class_sig, members) in enumerate(classes_dict.items()):
        for word in members:
            words.append(word)
            brown_class_indices.append(class_idx)
            # Get embedding
            vectors.append(ft_model.get_word_vector(word))
            
    X = np.array(vectors)
    
    # --- PCA Projection ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # --- K-Means Re-clustering ---
    # We want 15 clusters as per assignment instructions
    n_clusters = 15
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
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
    plot_scatter(axes[0], brown_class_indices, f'Original Brown Clusters ({lang})', 'tab20')
    
    # Plot 2: Colors based on K-Means (Re-clustered)
    plot_scatter(axes[1], kmeans_labels, f'K-Means Re-clustering on Embeddings ({lang})', 'tab20')
    
    plt.tight_layout()
    plt.show()

# Generate plots for each language
for lang in langs:
    print(f"\nGenerating plots for: {lang.upper()}")
    
    # Retrieve the clustering results we stored earlier
    classes_dict = clustering_results[lang]
    
    # Pass the CLASSES, not the unigrams
    process_and_visualize(lang, classes_dict, ft_models[lang])