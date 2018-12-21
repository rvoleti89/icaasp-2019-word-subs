import importlib
import argparse
import pickle
import os
from pathlib import Path

# Path to glove vectors txt file from Stanford NLP, creates this directory if not specified
GLOVE_PATH = os.path.expanduser('~/asr_simulator_data/glove')

load_dict = importlib.import_module('.data.01_load_cmu_and_features', 'src')
load_glove = importlib.import_module('.data.02_load_glove_vectors', 'src')
load_probs = importlib.import_module('.tools.03_compute_phono_distance', 'src')
corrupt_sentences = importlib.import_module('.tools.04_corrupt_sentences', 'src')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', dest='text_file', type=str, help='Path to text file containing '
                                                               'all sentences to be corrupted')
    parser.add_argument('-e', dest='wer', default=0.30, type=float, help='Word Error Rate (WER) for desired output')
    parser.add_argument('--redo', dest='redo', default=False, type=bool, help='Set to "True" if you want to overwrite'
                                                                              'the pickle file for the confusion'
                                                                              'word substitution DataFrame for a new'
                                                                              'corpus.')
    parser.add_argument('--glove', dest='glove_path', default=GLOVE_PATH, type=str, help='Directory which contains '
                                                                                         'the GloVe vectors txt file '
                                                                                         'from Stanford NLP. Default '
                                                                                         'will be created and file '
                                                                                         'will be downloaded if it'
                                                                                         'does not exist.')

    args = parser.parse_args()

    corpus_file = args.text_file
    corpus_file_no_ext = corpus_file.split('/')[-1].split('.')[0]
    wer = args.wer
    redo = args.redo
    glove_path = args.glove_path

    # Check if probs DataFrame pickle file already exists, if not, go through steps to generate it and load it
    if not Path(f'src/models/{corpus_file_no_ext}_word_substitution_df.pkl').is_file() or redo:
        # If this pickle doesn't exist or we want to recompute, we need to check for and/or generate the following:
        # 1. Phono Edit Distance pickle, 2. GloVe vectors gensim model,
        # 3. dict.pkl containing features and cmu_dictionary

        if not Path(glove_path).is_dir():
            print(f'Creating directory for GloVe model at {glove_path}')
            os.makedirs(glove_path)
            glove = load_glove.load_glove_model(glove_path)

    # Else, if pickle for word substitution model exists for a corpus, simply load and use to corrupt the corpus
    else:
        with open(f'src/models/{corpus_file_no_ext}_word_substitution_df.pkl', 'rb') as f:
            probs = pickle.load(f)

    with open(corpus_file, 'r') as f:
        corpus = f.read()
    # Use stop word list from NLTK, in script 03
    STOP = load_probs.STOP

    # Compute corrupted corrupted corpus for given WER
    corrupted_corpus = corrupt_sentences.replace_error_words(probs, corpus, error_rate=wer)

    with open(f'data/processed/corrupted_{corpus_file}', 'w') as f:
        f.write(corrupted_corpus)
