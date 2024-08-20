from hazm import Normalizer
from tqdm import tqdm
import config
import pandas as pd


def normalize_text(token: str):
    return Normalizer().normalize(text=token)


def convert_CoNLL_to_token_label(
        path: str,
        sep='\t',
) -> ([[str]], [[str]]):
    """
    convert IOB2 data format for token classification tasks to tokens and labels

    :return: a tuple containing two lists of list of strings
     which are tokens and labels for each sentence
    """
    token_seq, label_seq = [], []
    with open(path, mode='r', encoding='utf-8') as file:
        tokens, labels = [], []
        file = tqdm(file)
        for line in file:
            if line != '\n':
                token, label = line.strip().split(sep)
                token = token.lower()
                # token = normalize_text(token)
                tokens.append(token)
                labels.append(label)
            else:
                if len(tokens) > 0:
                    token_seq.append(tokens)
                    label_seq.append(labels)
                tokens, labels = [], []
    return token_seq, label_seq


def format_columns(tokens: [[str]], labels: [[str]]) -> pd.DataFrame:
    """
    Converts tokens and labels into a pandas DataFrame with columns 'tokens' and 'labels'.
    Labels are converted to their corresponding indices using the LABEL2IDX dictionary.
    """
    token_column = tokens
    label_column = [[config.LABEL2IDX[label] for label in label_list] for label_list in labels]
    data = {
        'tokens': token_column,
        'labels': label_column
    }
    return pd.DataFrame(data)

# tokens, labels = convert_CoNLL_to_token_label("path_to_your_conll_file.txt")
# df = format_columns(tokens, labels)
# print(df)
x = ['data/Persian-NER-part1.txt', 'data/Persian-NER-part2.txt', 'data/Persian-NER-part3.txt', 'data/Persian-NER-part4.txt', 'data/Persian-NER-part5.txt']
dfs = []
file_paths = x
# Process each file and append the resulting DataFrame to the list
for file_path in file_paths:
    tokens, labels = convert_CoNLL_to_token_label(file_path)
    df = format_columns(tokens, labels)
    dfs.append(df)
# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dfs, ignore_index=True)