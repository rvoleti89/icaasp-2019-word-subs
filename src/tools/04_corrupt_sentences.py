import re
import pickle
import numpy as np
import importlib

phono_module = importlib.import_module('.tools.03_compute_phono_distance', 'src')
Sentence = phono_module.Sentence
STOP = phono_module.STOP


def identify_subs(tokens):
    x = [(word, np.random.rand()) for word in tokens]
    return x


def pick_similar_word(prob_df):
    try:
        word = (prob_df.sample(n=1, weights=prob_df.Probability)).index.remove_unused_levels().levels[1][0]
    except:
        return ''
    return word


def replace_error_words(probs_df, sentence, error_rate):
    sentence_class = Sentence(sentence)

    tokens = sentence_class.tokens_cased

    words_to_replace = identify_subs(tokens)

    corrupt_sent = []
    for word, num in words_to_replace:
        if word in probs_df.index.levels[0] and num < error_rate:  # replace word
            probs = probs_df.loc[(word, slice(None)), :]
            corrupt_sent.append(pick_similar_word(probs))
        else:
            corrupt_sent.append(word)

    sentence_class.tokens_cased = corrupt_sent
    # words_only = [word for word, prob in words_to_replace]

    # probs = probs_df.loc[(words_only, slice(None)), :]
    #
    # probs_group = probs.groupby(level='Corpus Word')
    # sim_words_dict = probs_group.apply(pick_similar_word).to_dict()

    # for k, v in list(sim_words_dict.items()):
    #    sentence_class.tokens_cased = [re.sub(r"\b" + k + r"\b", v, w) for w in sentence_class.tokens_cased]

    sentence_class.tokens = [w.lower() for w in sentence_class.tokens_cased]
    sentence_class.tokens_without_stop = [w for w in sentence_class.tokens if w not in STOP]
    sentence_class.tokens_cased_without_stop = [w for w in sentence_class.tokens_cased if w not in STOP]

    sentence_class.raw = " ".join(sentence_class.tokens_cased)

    return sentence_class.raw


if __name__ == '__main__':
    sample_sentence = 'ice cream is the best dessert in the world as we know it is it is it is it is.'
    wer = 0.50
    with open('../models/test_probs_df.pkl', 'rb') as f:
        probs = pickle.load(f)

    print(replace_error_words(probs, sample_sentence, error_rate=wer))
