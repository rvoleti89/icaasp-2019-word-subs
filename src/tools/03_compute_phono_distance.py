import pickle
import numpy as np
import pandas as pd
import nltk
import importlib
import os
from corpustools.symbolsim import phono_edit_distance
from pathlib import Path

# Global variables
STOP = set(nltk.corpus.stopwords.words("english"))
MAX_WORDS = 15  # Set maximum number of confusion words for each word
PHONO_THRESHOLD = 60  # Set threshold for maximum phonological edit distance for confusion words


class Sentence:
    def __init__(self, sentence):
        """
        Takes a sentence or string of multiple words and tokenizes in 4 different ways:
            - all lower case
            - all lower case with stop words removed
            - keep case
            - keep case with stop words removed
        :param sentence: a string of words
        """
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]
        self.tokens_cased = [t for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_cased_without_stop = [t for t in self.tokens_cased if t not in STOP]


def find_distances(corpus, phono_dict, features, model, use_stoplist=False, n=1000):
    class_text = Sentence(corpus)

    # Convert dict keys and word list to upper case for comparison purposes
    phono_words = [word.upper() for word in list(phono_dict.keys())]

    # Create set of all unique tokens in the corpus
    if use_stoplist:
        set_of_unique_tokens = set(class_text.tokens_cased_without_stop)
    else:
        set_of_unique_tokens = set(class_text.tokens_cased)

    # Filter set of unique tokens to only those that are in the model and the phonological corpus
    set_of_unique_tokens = {token for token in set_of_unique_tokens if
                            token in model and token.upper() in phono_words}

    # Empty lists for word tuples (index) and distances (column) of resulting DataFrame
    distances = []
    word_tuples = []

    for token in set_of_unique_tokens:
        similar_set = model.most_similar(token, topn=n)

        # Only keep words and filter
        similar_set = [item[0] for item in similar_set]
        similar_set = [sim_word for sim_word in similar_set if
                       sim_word in model and sim_word.upper() in phono_words
                       and token.upper() != sim_word.upper()]

        for sim_word in similar_set:
            distance = phono_edit_distance.phono_edit_distance(phono_dict.get(token.upper()),
                                                               phono_dict.get(sim_word.upper()),
                                                               'transcription',
                                                               features
                                                               )
            word_tuples.append((token, sim_word))
            distances.append(distance)

    index = pd.MultiIndex.from_tuples(tuples=word_tuples, names=['Corpus Word', 'Similar Word from Model'])
    columns = ['Phono Edit Distance']

    dist_frame = pd.DataFrame(distances, index=index, columns=columns)

    return dist_frame


def max_confusion_words(distances, n_words=MAX_WORDS):
    if len(distances) > n_words:
        return distances.nsmallest(n_words, 'Phono Edit Distance')
    else:
        return distances


def filter_distances_df(distances_df, phono_threshold=PHONO_THRESHOLD):
    filtered = distances_df.loc[distances_df['Phono Edit Distance'] < phono_threshold]
    dist_group = filtered.groupby(level='Corpus Word', group_keys=False)
    distances_filtered = dist_group.apply(max_confusion_words).groupby(level='Corpus Word', group_keys=False)
    return distances_filtered


def dist_to_prob(dist, sigma, total):
    prob = np.exp(-1 * np.square(dist) / (2*np.square(sigma))) / total
    return prob


def find_probabilities(df):
    df2 = df.copy()

    # Set sigma
    sigma = np.mean(df2['Phono Edit Distance'].values)

    total = np.sum(np.exp(-1 * np.square(df2['Phono Edit Distance'].values) / (2 * np.square(sigma))))
    df2['Probability'] = df2['Phono Edit Distance'].apply(lambda x: dist_to_prob(x, sigma, total))
    return df2


if __name__ == '__main__':
    glove_module = importlib.import_module('.data.02_load_glove_vectors', 'src')
    glove = glove_module.load_glove_model(os.path.expanduser('~/asr_simulator_data/glove'))

    with open('../models/dict.pkl', 'rb') as f:
        cmu_dict, arpabet2hayes = pickle.load(f)

    test_corpus = "The quick brown fox jumped over the lazy dogs in Syria. " \
                  "Cheesecake is the best dessert in the world as we know it, but not ice cream."

    if not Path('../models/glove_hayes_all_test.pkl').is_file():
        distances = find_distances(test_corpus, cmu_dict, arpabet2hayes, model=glove, use_stoplist=False)
        with open('../models/glove_hayes_all_test.pkl', 'wb') as f:
            pickle.dump(distances, f)
    else:
        with open('../models/glove_hayes_all_test.pkl', 'rb') as f:
            distances = pickle.load(f)

    distances_filtered = filter_distances_df(distances)
    probs = distances_filtered.apply(find_probabilities)

    # Set display options
    pd.options.display.float_format = '{:,.4f}'.format
    probs = probs.sort_values(['Corpus Word', 'Probability'], ascending=[True, False])

    with open('../models/test_probs_df.pkl', 'wb') as f:
        pickle.dump(probs, f)

    print('Word of interest: "fox"')
    print(probs.loc['Cheesecake'])