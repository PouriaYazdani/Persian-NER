
DATA_PART1 = 'data/Persian-NER-part1.txt'
DATA_PART2 = 'data/Persian-NER-part2.txt'
DATA_PART3 = 'data/Persian-NER-part3.txt'
DATA_PART4 = 'data/Persian-NER-part4.txt'
DATA_PART5 = 'data/Persian-NER-part5.txt'

LABEL2IDX = {
    'O': 0,
    'B-DAT': 1,
    'B-PER': 2,
    'B-ORG': 3,
    'B-LOC': 4,
    'B-EVE': 5,
    'I-DAT': 6,
    'I-PER': 7,
    'I-ORG': 8,
    'I-LOC': 9,
    'I-EVE': 10
}

IDX2LABEL = {i: k for k, i in LABEL2IDX.items()}

CLS = [101]
SEP = [102]
VALUE_TOKEN = [0]
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
EPOCHS = 5
NUM_CLASS = 11
LEARNING_RATE = 5e-5