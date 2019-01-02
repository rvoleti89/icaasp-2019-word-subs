import src.models
import importlib
import argparse
import pickle
import os
from pathlib import Path
from corpustools.corpus import io

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
    parser.add_argument('-o', dest='output_loc', default=None, type=str,
                        help='Path to directory for corrupted output text file')
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
    corpus_file = os.path.expanduser(args.text_file)
    if args.output_loc is not None:
        output_dir = os.path.expanduser(args.output_loc)
    else:
        output_dir = os.path.dirname(corpus_file)
    corpus_file_no_ext = corpus_file.split('/')[-1].split('.')[0]
    wer = args.wer
    redo = args.redo
    glove_path = os.path.expanduser(args.glove_path)

    # Read file into variable
    with open(corpus_file, 'r') as f:
        corpus = f.readlines()

    # Indicate directory where models are stored
    models_path = src.models.__path__[0]

    # Check if probs DataFrame pickle file already exists, if not, go through steps to generate it and load it
    if not Path(os.path.join(models_path, f'{corpus_file_no_ext}_word_substitution_df.pkl')).is_file() or redo:
        # If this pickle doesn't exist or we want to recompute, we need to check for and/or generate the following:
        # 1. Phono Edit Distance pickle, 2. GloVe vectors gensim model,
        # 3. dict.pkl containing features and cmu_dictionary

        if not Path(os.path.join(models_path, f'{corpus_file_no_ext}_glove_hayes_phono_dist.pkl')).is_file():
            if not Path(glove_path).is_dir():
                print(f'Creating directory for GloVe model at {glove_path} and saving gensim model for GloVe'
                      f' vectors to {models_path}')
                os.makedirs(glove_path)
            glove = load_glove.load_glove_model(glove_path, os.path.join(models_path, 'glove'))

            # Check if dict.pkl exists, create and save if not
            if not Path(os.path.join(models_path, 'dict.pkl')).is_file():
                data_path = os.path.expanduser('~/asr_simulator_data/')

                # Check if arpabet2hayes feature vector binary exists, download, and load
                if not Path(os.path.join(data_path, 'arpabet2hayes')).is_file():
                    try:
                        io.binary.download_binary('arpabet2hayes', os.path.join(data_path, 'arpabet2hayes'))
                        print(f'Downloaded arpabet2hayes binary file for Hayes feature matrix to {data_path}')
                    except OSError as e:
                        print(f'Creating data directory in home folder to save files at {data_path}')
                        os.makedirs(data_path)
                        print(f'Downloading arpabet2hayes binary file for Hayes feature matrix to {data_path}')
                        io.binary.download_binary('arpabet2hayes', os.path.join(data_path, 'arpabet2hayes'))

                # Download binary for Hayes set of features from arpabet transcriptions
                arpabet2hayes = io.binary.load_binary(os.path.join(data_path, 'arpabet2hayes'))

                # Check if CMU dict file exists, if not download it and load
                if not Path(os.path.join(data_path, 'cmudict.0.7a_SPHINX_40')).is_file():
                    print(f'Downloading CMU Pronouncing Dictionary (ARPABET) to {data_path}')
                    load_dict.download_cmu(data_path)

                cmu_dict = load_dict.read_cmu(os.path.join(data_path, 'cmudict.0.7a_SPHINX_40'))

                # Save pickle file for cmu_dict and arpabet2hayes for easier loading next time
                with open(os.path.join(models_path, 'dict.pkl'), 'wb') as f:
                    pickle.dump([cmu_dict, arpabet2hayes], f)
            else:
                with open(os.path.join(models_path, 'dict.pkl'), 'rb') as f:
                    cmu_dict, arpabet2hayes = pickle.load(f)
            num_words = len(cmu_dict)
            print(f'Loaded CMU Pronouncing Dictionary with ARPABET transcriptions for {num_words} English words '
                  f'and feature matrix by Hayes.')
            print(f'Computing phonological edit distances for all similar words to unique words in {corpus_file}',
                  '\nThis may take a while...')
            distances = load_probs.find_distances(''.join(corpus), cmu_dict, arpabet2hayes, model=glove,
                                                  use_stoplist=False)
            print('Done!')
            # Save pickle for phono edit distances
            pkl_path = os.path.join(models_path, f'{corpus_file_no_ext}_glove_hayes_phono_dist.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(distances, f)
            print(f'Pickle file saved at {pkl_path}')

        else:  # If phono distance pickle exists
            with open(os.path.join(models_path, f'{corpus_file_no_ext}_glove_hayes_phono_dist.pkl'), 'rb') as f:
                distances = pickle.load(f)

        distances_filtered = load_probs.filter_distances_df(distances)
        probs = distances_filtered.apply(load_probs.find_probabilities)

        # Save probability DataFrame pickle for word substitutions
        word_pkl_path = os.path.join(models_path, f'{corpus_file_no_ext}_word_substitution_df.pkl')
        with open(word_pkl_path, 'wb') as f:
            pickle.dump(probs, f)

        print(f'Saved word substitution DataFrame pickle file at {word_pkl_path}.')

    # Else, if pickle for word substitution model exists for a corpus, simply load and use to corrupt the corpus
    else:
        print(f'Loading word substitution DataFrame for {corpus_file}')
        with open(os.path.join(models_path, f'{corpus_file_no_ext}_word_substitution_df.pkl'), 'rb') as f:
            probs = pickle.load(f)

    # Compute corrupted corrupted corpus for given WER
    # corrupted_corpus = corrupt_sentences.replace_error_words(probs, corpus, error_rate=wer)
    corrupted_corpus = []
    for sent in corpus:
        corrupted_corpus.append(corrupt_sentences.replace_error_words(probs, sent, error_rate=wer) + '\n')

    corrupted_file_with_ext = corpus_file.split('/')[-1]
    with open(os.path.join(output_dir, f'corrupted_{corrupted_file_with_ext}'), 'w') as f:
        f.writelines(corrupted_corpus)

    print(f'Corrupted {corpus_file} with a WER of {wer*100}% and saved the output as',
          os.path.join(output_dir, f'corrupted_{corrupted_file_with_ext}'))
