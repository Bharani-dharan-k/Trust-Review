import pandas as pd
import numpy as np
langs = ['Hindi', 'Punjabi', 'Assamese', 'Bengali', 'Gujarati', 'Kannada', 'Marathi', 'Tamil', 'Malayalam', 'Odia', 'Telugu', 'Bodo', 'Urdu']

codes = ['hi', 'pa', 'as', 'bn', 'gu', 'kn', 'mr', 'ta', 'ml', 'or', 'te', 'bd', 'ur']

metadata = pd.read_csv('metadata.tsv', sep='\t')
idx = np.random.permutation(metadata.index)

# for lang, code in zip(langs, codes):
for lang in codes:
    with open(f"{lang}.txt", "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    metadata = pd.read_csv('metadata.tsv', sep='\t')
    metadata.columns = ['GENERIC CATEGORIES', 'CATEGORY', 'SUB-CATEGORY', 'PRODUCT', 'BRAND', 'ASPECTS', 'ASPECT COMBO', 'ENGLISH REVIEW', 'LABEL']
    metadata['INDIC REVIEW'] = lines

    metadata = metadata.reindex(idx)

    metadata[:1000].to_json(f"test/{lang}.json", orient='records', lines=True)
    metadata[1000:].to_json(f"dev/{lang}.json", orient='records', lines=True)

    # metadata.to_json(f"dataset/{code}.json", orient='records', lines=True)