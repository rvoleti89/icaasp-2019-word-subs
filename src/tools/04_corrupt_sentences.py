import re
import pickle
import numpy as np
import importlib


def identify_subs(tokens, error_rate):
    x = {word: np.random.rand() for word in tokens}
    filtered = {k: v for k, v in x.items() if v < error_rate}

    # Return randomly selected tokens list
    return list(filtered.keys())


def pick_similar_word(prob_df):
    try:
        word = (prob_df.sample(n=1, weights=prob_df.Probability)).index.remove_unused_levels().levels[1][0]
    except:
        return ''
    return word


def replace_error_words(probs_df, sentence, error_rate):
    tokens = sentence.tokens_cased

    words_to_replace = identify_subs(tokens, error_rate=error_rate)

    probs = probs_df.loc[(words_to_replace, slice(None)), :]

    probs_group = probs.groupby(level='Corpus Word')

    sim_words_dict = probs_group.apply(pick_similar_word).to_dict()

    for k, v in list(sim_words_dict.items()):
        sentence.tokens_cased = [re.sub(r"\b" + k + r"\b", v, w) for w in sentence.tokens_cased]

    sentence.tokens = [w.lower() for w in sentence.tokens_cased]
    sentence.tokens_without_stop = [w for w in sentence.tokens if w not in STOP]
    sentence.tokens_cased_without_stop = [w for w in sentence.tokens_cased if w not in STOP]

    sentence.raw = " ".join(sentence.tokens_cased)

    return sentence.raw


if __name__ == '__main__':
    sample_sentence = 'The quick brown fox jumped over the lazy dogs in Syria.'
    wer = 0.30
    phono_module = importlib.import_module('.tools.03_compute_phono_distance', 'src')
    sample_sentence_class = phono_module.Sentence(sample_sentence)
    STOP = phono_module.STOP

    with open('../models/test_probs_df.pkl', 'rb') as f:
        probs = pickle.load(f)

    print(replace_error_words(probs, sample_sentence_class, error_rate=wer))
