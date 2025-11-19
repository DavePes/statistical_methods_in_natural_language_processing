#pip install datasets
from datasets import load_dataset
#langs = ['cs', 'en', 'de']
langs = ['cs']
datasets = {}
for lang in langs:
    dataset = load_dataset("ufal/npfl147", lang)
    datasets[lang] = dataset

#print(cz, en, ge)

texts = [[x['text'] for x in data['train']] for data in datasets.values()]


## tokenizer
from sacremoses import MosesTokenizer
tokens = {}
for i, lang in enumerate(langs):
    tok = MosesTokenizer(lang=lang)
    lang_tokens = tok.tokenize(texts[i])
    tokens[lang] = lang_tokens

## counting

from collections import Counter
def dataset_counter(tokens):
    data_sizes = {}
    unigrams_frequencies,unigrams_counts = {}, {}
    bigrams_frequencies,bigrams_counts = {}, {}
    trigrams_frequencies,trigrams_counts = {}, {}
    for lang in langs:
        bigram = Counter()
        trigram = Counter()
        for i in range(len(tokens[lang])-1):
            bigram[(tokens[lang][i], tokens[lang][i+1])] += 1
            if (i < len(tokens[lang])-2):
                trigram[(tokens[lang][i], tokens[lang][i+1], tokens[lang][i+2])] += 1
        unigrams_frequencies[lang] = Counter(tokens[lang])
        unigrams_counts[lang] = len(unigrams_frequencies[lang])
        bigrams_frequencies[lang] = bigram
        bigrams_counts[lang] = len(bigrams_frequencies[lang])
        trigrams_frequencies[lang] = trigram
        trigrams_counts[lang] = len(trigrams_frequencies[lang])

        data_sizes[lang] = len(tokens[lang])
    return (unigrams_frequencies,unigrams_counts,
            bigrams_frequencies,bigrams_counts,
            trigrams_frequencies,trigrams_counts,
            data_sizes)
    
(unigrams_frequencies,unigrams_counts,
 bigrams_frequencies,bigrams_counts,
 trigrams_frequencies,trigrams_counts,
 data_sizes) = dataset_counter(tokens)

## entropy
from math import log2
from tabulate import tabulate
def bigram_entropy(bigrams_frequencies, unigrams_frequencies, data_sizes):
    entropies = {}
    table_data = []
    for lang in langs:
        entropies[lang] = 0
        for w1, w2 in bigrams_frequencies[lang]:
            p_w1_w2 = bigrams_frequencies[lang][(w1, w2)] / data_sizes[lang]
            p_w2_given_w1 = bigrams_frequencies[lang][(w1, w2)] / unigrams_frequencies[lang][w1]
            entropies[lang] -= p_w1_w2 * log2(p_w2_given_w1)
        row = [
            lang.upper(),
            unigrams_counts[lang],
            bigrams_counts[lang],
            data_sizes[lang],
            f"{entropies[lang]:.4f}" # Format entropy to 4 decimal places
        ]
        table_data.append(row)
    return table_data
table_data = bigram_entropy(bigrams_frequencies, unigrams_frequencies, data_sizes)
print(tabulate(table_data, headers=["Language", "Unigrams", "Bigrams", "Data Size", "Bigram Entropy"]))
# explain differences in entropies



## same but XLM-R tokenizer
from transformers import XLMRobertaTokenizer
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
xlm_tokens = {}
for i, lang in enumerate(langs):
    lang_tokens = xlm_tokenizer.tokenize(texts[i])
    xlm_tokens[lang] = lang_tokens

(unigrams_frequencies,unigrams_counts,
 bigrams_frequencies,bigrams_counts,
 trigrams_frequencies,trigrams_counts,
 data_sizes) = dataset_counter(xlm_tokens)

table_data = bigram_entropy(bigrams_frequencies, unigrams_frequencies, data_sizes)
print(tabulate(table_data, headers=["Language", "Unigrams", "Bigrams", "Data Size", "Bigram Entropy (XLM-R)"]))

## dataset splits

tokens_train = {}
tokens_val = {}
tokens_test = {}

for lang in langs:
    total_tokens = len(tokens[lang])
    
    # Example split: 80% Train, 10% val, 10% Test
    train_end = 700
    val_end = train_end + 200
    
    tokens_train[lang] = tokens[lang][:train_end]
    tokens_val[lang] = tokens[lang][train_end:val_end]
    tokens_test[lang] = tokens[lang][val_end:]


## counting


(train_uni_freq, train_uni_count,
 train_bi_freq, train_bi_count,
 train_tri_freq, train_tri_count,
 train_data_sizes) = dataset_counter(tokens_train)

(test_uni_freq, test_uni_count,
 test_bi_freq, test_bi_count,
 test_tri_freq, test_tri_count,
 test_data_sizes) = dataset_counter(tokens_test)

val_stats = dataset_counter(tokens_val)

test_stats = (test_uni_freq, test_uni_count, test_bi_freq, test_bi_count, test_tri_freq, test_tri_count, test_data_sizes)
train_stats = (train_uni_freq, train_uni_count, train_bi_freq, train_bi_count, train_tri_freq, train_tri_count, train_data_sizes)   

## entropy ## lambdas


def compute_probs(stats):
    uni_freq, uni_count, bi_freq, bi_count, tri_freq, tri_count, data_sizes = stats
    unigram_probs = {}
    bigram_probs = {}
    trigram_probs = {}

    for lang in langs:
        unigram_probs[lang] = {}
        bigram_probs[lang] = {}
        trigram_probs[lang] = {}
        for w1 in uni_freq[lang]:
            p_w1 = uni_freq[lang][w1] / data_sizes[lang]
            unigram_probs[lang][w1] = p_w1
        for w1, w2 in bi_freq[lang]:
            p_w2_given_w1 = bi_freq[lang][(w1, w2)] / uni_freq[lang][w1]
            bigram_probs[lang][(w1, w2)] = p_w2_given_w1
        for w1, w2, w3 in tri_freq[lang]:
            p_w3_given_w1_w2 = tri_freq[lang][(w1, w2, w3)] / bi_freq[lang][(w1, w2)]
            trigram_probs[lang][(w1, w2, w3)] = p_w3_given_w1_w2
    return unigram_probs, bigram_probs, trigram_probs

def interpolated_cross_entropy(tokens,unigram_probs,bigram_probs,trigram_probs, lambdas=[0.25, 0.25, 0.25, 0.25]):
    """
    Calculates cross-entropy on TEST data using probabilities from TRAIN data.
    """
    entropies = []
    l3, l2, l1, l0 = lambdas
    for lang in langs:
        log_prob_sum = 0
        N = len(tokens_test[lang])
        
        # Vocabulary size (V) estimated from training unigrams for Uniform prob
        V = len(unigram_probs[lang])
        for i in range(2, N):
            w1 = tokens[lang][i-2]
            w2 = tokens[lang][i-1]
            w3 = tokens[lang][i]
            p_tri = trigram_probs[lang].get((w1, w2, w3), 0.0)
            p_bi  = bigram_probs[lang].get((w2, w3), 0.0)
            p_uni = unigram_probs[lang].get(w3, 0.0)
            p_zero = 1 / V
            # linear transformation
            p_combined = (l3 * p_tri) + (l2 * p_bi) + (l1 * p_uni) + (l0 * p_zero)
            if p_combined > 0:
                log_prob_sum -= log2(p_combined)
            else:
                print(f"Zero probability for trigram ({w1}, {w2}, {w3}) in language {lang}")
                pass
        entropy = log_prob_sum / (N - 2)
        entropies.append((lang, entropy))
    return entropies

## def best lambdas EM algorithm
from math import abs
def em_optimal_lambdas(tokens, unigram_probs, bigram_probs, trigram_probs, default_lambdas=[0.25, 0.25, 0.25, 0.25]):
    lambdas = default_lambdas
    lambdas_new = [100, 100, 100, 100]
    unigram_probs, bigram_probs, trigram_probs = compute_probs(val_stats)

    while 
    
    return lambdas

        







    
