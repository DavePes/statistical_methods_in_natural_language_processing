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
data_sizes = {}
def dataset_counter(tokens):
    unigrams_frequencies,unigrams_counts = {}, {}
    bigrams_frequencies,bigrams_counts = {}, {}
    trigrams_frequencies,trigrams_counts = {}, {}
    for lang in langs:
        unigram = Counter()
        bigram = Counter()
        trigram = Counter()
        for i in range(len(tokens[lang])-1):
            bigram[(tokens[lang][i], tokens[lang][i+1])] += 1
            if (i < len(tokens[lang])-2):
                trigram[(tokens[lang][i], tokens[lang][i+1], tokens[lang][i+2])] += 1
        unigrams_frequencies[lang] = unigram(tokens[lang])
        unigrams_counts[lang] = len(unigram)
        bigrams_frequencies[lang] = bigram
        bigrams_counts[lang] = len(bigram)
        trigrams_frequencies[lang] = trigram
        trigrams_counts[lang] = len(trigram)

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
    lang_tokens = []
    for text in texts[i]:
        tokens_ids = xlm_tokenizer.encode(text, add_special_tokens=False)
        tokens_str = [xlm_tokenizer.convert_ids_to_tokens(tid) for tid in tokens_ids]
        lang_tokens.extend(tokens_str)
    xlm_tokens[lang] = lang_tokens




    

    
