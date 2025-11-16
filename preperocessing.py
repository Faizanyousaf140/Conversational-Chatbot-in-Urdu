import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import sentencepiece as spm
import os
# Read the dataset
df = pd.read_csv('final_main_dataset.tsv', sep='\t', encoding='utf-8')

print(f"Dataset loaded successfully!")

# Breif Eda on Dataset
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few sentences:")
print(df['sentence'].head())

def normalize_urdu_text(text):

    if pd.isna(text):  # Handle NaN values
        return ""

    text = str(text)

    # remove common urdu diacritics ie standardize alef and yeh forms(zabar, zer, pesh, etc.)
    # Some most used and common standardized
    diacritics = [
        '\u064B',
        '\u064C',
        '\u064D',
        '\u064E',
        '\u064F',
        '\u0650',
        '\u0651',
        '\u0652',
        '\u0653',
        '\u0654',
        '\u0655',
        '\u0656',
        '\u0657',
        '\u0658',
    ]

    for diacritic in diacritics:
        text = text.replace(diacritic, '')

    # Standardize different forms of alef
    text = text.replace('أ', 'ا')
    text = text.replace('إ', 'ا')
    text = text.replace('آ', 'ا')
    text = text.replace('ٱ', 'ا')

    # Standardize different forms of yeh
    text = text.replace('ی', 'ی')
    text = text.replace('ے', 'ے')
    text = text.replace('ئ', 'ی')

    # Standardize heh forms
    text = text.replace('ۃ', 'ہ')
    text = text.replace('ھ', 'ہ')

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text.strip()

# Test the function
test_sentence = "کبھی کبھار ہی خیالی پلاو بناتا ہوں"
normalized = normalize_urdu_text(test_sentence)
print(f"Original:   {test_sentence}")
print(f"Normalized: {normalized}")
print("\nNormalization function ready!")


# Apply Normalization to All Sentences

df['normalized_sentence'] = df['sentence'].apply(normalize_urdu_text)

# Remove empty sentences after normalization
df = df[df['normalized_sentence'].str.len() > 0].reset_index(drop=True)

print(f"Normalization complete!")

print(f"Total sentences after cleaning: {len(df)}")
print("\nSample normalized sentences:")
for i in range(5):
    print(f"{i+1}. {df['normalized_sentence'].iloc[i]}")


# Save Normalized Sentences to Text File for tokenizer training

with open('urdu_sentences.txt', 'w', encoding='utf-8') as f:
    for sentence in df['normalized_sentence']:
        f.write(sentence + '\n')
print(f"Saved {len(df)} normalized sentences to 'urdu_sentences.txt'")

# Train SentencePiece Tokenizer with Automatic Vocab Size

print("Training SentencePiece Unigram tokenizer...")

# Checking unique characters
with open('urdu_sentences.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    unique_chars = len(set(text))

print(f"Unique characters in dataset: {unique_chars}")

# Set Reasonable vocab size based on dataset size
total_sentences = len(df)

if total_sentences < 5000:
    vocab_size = 3000
elif total_sentences < 10000:
    vocab_size = 4000
elif total_sentences < 15000:
    vocab_size = 5000
else:
    vocab_size = 6000

print(f"Setting vocabulary size to: {vocab_size}")
print("Training tokenizer...\n")

# Train the tokenizer with adjusted vocab size
spm.SentencePieceTrainer.train(
    input='urdu_sentences.txt',
    model_prefix='urdu_tokenizer',
    vocab_size=vocab_size,
    character_coverage=1.0,  # Cover all Urdu characters
    model_type='unigram',  # Unigram Language Model
    pad_id=0,  # Padding token
    unk_id=1,  # Unknown token
    bos_id=2,  # Begin of sentence
    eos_id=3,  # End of sentence
    user_defined_symbols=['[MASK]'],  # Masking special tokens
    num_threads=4,
)

print("Tokenizer trained successfully!")
print(f"Vocabulary size: {vocab_size}")
print("Files created: urdu_tokenizer.model, urdu_tokenizer.vocab")


# Load the trained tokenizer
sp = spm.SentencePieceProcessor()
sp.load('urdu_tokenizer.model')
print(f"Vocabulary size: {sp.vocab_size()}")
print(f"\n--- Testing Tokenizer ---")

# Test sentences
test_sentences = [
    "کبھی کبھار ہی خیالی پلاو بناتا ہوں",
    "آج موسم بہت اچھا ہے",
    "بولو کیسے ہو"
]

for sent in test_sentences:
    tokens = sp.encode_as_pieces(sent)
    ids = sp.encode_as_ids(sent)

    print(f"\nOriginal:  {sent}")
    print(f"Tokens:    {tokens}")
    print(f"Token IDs: {ids[:10]}...")  # Show first 10 IDs

    # Decode back
    decoded = sp.decode_pieces(tokens)
    print(f"Decoded:   {decoded}")

print("\nTokenizer working correctly!")

# Tokenize All Sentences

print("Tokenizing all sentences...")

# Tokenize all normalized sentences
df['tokenized'] = df['normalized_sentence'].apply(lambda x: sp.encode_as_ids(x))
df['token_count'] = df['tokenized'].apply(len)

print(f" All sentences tokenized!")
print(f"\nToken Statistics")
print(f"Average tokens per sentence: {df['token_count'].mean():.2f}")
print(f"Min tokens: {df['token_count'].min()}")
print(f"Max tokens: {df['token_count'].max()}")
print(f"Median tokens: {df['token_count'].median():.2f}")

# Show distribution
print(f"\nToken Length Distribution ")
print(df['token_count'].describe())

#  Split Dataset (80% Train, 10% Val, 10% Test)

print("Splitting dataset...")

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Second split: 10% val, 10% test (from the 20% temp)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f" Dataset split complete!")
print(f"\nSplit Statistics")
print(f"Training set:   {len(train_df)} sentences ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val_df)} sentences ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set:       {len(test_df)} sentences ({len(test_df)/len(df)*100:.1f}%)")
print(f"Total:          {len(df)} sentences")

#  Save Processed Datasets

print("Saving processed datasets...")

# Save to CSV files
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print(" Saved datasets:")
print("  train_data.csv")
print("  val_data.csv")
print("  test_data.csv")

# Save tokenized versions for further use
import pickle

with open('train_tokenized.pkl', 'wb') as f:
    pickle.dump(train_df['tokenized'].tolist(), f)

with open('val_tokenized.pkl', 'wb') as f:
    pickle.dump(val_df['tokenized'].tolist(), f)

with open('test_tokenized.pkl', 'wb') as f:
    pickle.dump(test_df['tokenized'].tolist(), f)

print("Saved tokenized versions (pickle files)")


# Summary and Verification

print("PREPROCESSING COMPLETE - SUMMARY")


print(f"\nDataset Statistics:")
print(f"   Total sentences: {len(df)}")
print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

print(f"\nTokenizer:")
print(f"   Vocabulary size: {sp.vocab_size()}")
print(f"   Model type: Unigram LM")
print(f"   Special tokens: <pad>, <unk>, <s>, </s>, [MASK]")

print(f"\n Files Created:")
files = [
    'urdu_tokenizer.model',
    'urdu_tokenizer.vocab',
    'train_data.csv',
    'val_data.csv',
    'test_data.csv',
    'train_tokenized.pkl',
    'val_tokenized.pkl',
    'test_tokenized.pkl'
]

for file in files:
    exists = "✓" if os.path.exists(file) else "✗"
    print(f"   {exists} {file}")

print(f"\n Sample from Training Set:")
sample = train_df.sample(3)
for idx, row in sample.iterrows():
    print(f"\n   Sentence: {row['normalized_sentence']}")
    print(f"   Tokens: {sp.encode_as_pieces(row['normalized_sentence'])[:8]}...")
    print(f"   Token count: {row['token_count']}")

print("\Phase 1 Preprocessing Complete!")