from pathlib import Path
import os
import pickle
import gensim
import wget
import zipfile
from gensim.scripts.glove2word2vec import glove2word2vec


def load_glove_model(path):
    """
    :param path: folder to which to download GloVe model from Stanford NLP
    :return: glove gensim KeyedVectors model
    """
    url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    if not Path(path).is_dir():
        os.makedirs(path)
        print(f'Creating directory for glove model at {path}')

    gensim_model_path = '../models/glove'
    gensim_model_abspath = os.path.abspath(gensim_model_path)
    if not Path(gensim_model_path).is_file():
        if not Path(os.path.join(path, 'glove.840B.300d.txt')).is_file():
            print(f'Downloading pre-trained GloVe Vectors (840B 300d vectors trained on Common Crawl) to {path}')
            zip_file = wget.download(url, os.path.join(path, 'glove.840B.300d.zip'))

            print(f'Unzipping {zip_file}')
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(path)

            print(f'Deleting {zip_file}')
            os.remove(zip_file)
        print(f'Generating and saving GloVe gensim model at {gensim_model_abspath}, this may take several minutes.')
        tmp_file = "/tmp/glove.840B.300d.w2v.txt"
        glove2word2vec(os.path.join(path, 'glove.840B.300d.txt'), tmp_file)
        glove = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
        glove.save(gensim_model_path)
        print(f'Loading GloVe vector gensim model from {gensim_model_abspath}.')
        os.remove(tmp_file)
    else:
        print(f'GloVe gensim KeyedVectors model exists! Loading model from {gensim_model_abspath}.')
        glove = gensim.models.KeyedVectors.load(gensim_model_path)
    return glove


if __name__ == '__main__':
    GLOVE_PATH = os.path.expanduser('~/asr_simulator_data/glove')
    if not Path(GLOVE_PATH).is_dir():
        print(f'Creating directory for glove model at {GLOVE_PATH}')
        os.makedirs(GLOVE_PATH)

    glove = load_glove_model(GLOVE_PATH)
