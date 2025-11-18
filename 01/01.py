#pip install datasets
from datasets import load_dataset

cz = load_dataset("ufal/npfl147", "cs")
#en = load_dataset("ufal/npfl147", "en")
#de = load_dataset("ufal/npfl147", "de")
#print(cz, en, ge)


cz_texts = [x['text'] for x in cz['train']]
x=1