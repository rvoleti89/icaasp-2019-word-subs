from pathlib import Path
import os
import gensim
import wget
import zipfile
from gensim.scripts.glove2word2vec import glove2word2vec


def load_glove_model(path, gensim_model_path=os.path.expanduser('~/asr_simulator_data/models')):
    """
    :param path: folder to which to download GloVe model from Stanford NLP
    :param gensim_model_path: path which contains gensim model for pre-trained GloVe vectors
    :return: glove gensim KeyedVectors model
    """
    url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    if not Path(path).is_dir():
        os.makedirs(path)
        print(f'Creating directory for glove model at {path}')

    gensim_model_abspath = os.path.abspath(gensim_model_path)
    if not Path(os.path.join(gensim_model_abspath, 'glove')).is_file():
        if not Path(os.path.join(path, 'glove.840B.300d.txt')).is_file():
            print(f'Downloading pre-trained GloVe Vectors (840B 300d vectors trained on Common Crawl) to {path}')
            zip_file = wget.download(url, os.path.join(path, 'glove.840B.300d.zip'))

            print(f'\nUnzipping {zip_file}')
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(path)

            print(f'Deleting {zip_file}')
            os.remove(zip_file)
        print(f'Generating and saving GloVe gensim model to {gensim_model_abspath}/glove, '
              f'this may take several minutes.')
        tmp_file = "/tmp/glove.840B.300d.w2v.txt"
        glove2word2vec(os.path.join(path, 'glove.840B.300d.txt'), tmp_file)
        glove = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
        try:
            glove.save(os.path.join(gensim_model_abspath, 'glove'))
        except OSError as e:
            print(f'Creating directory for gensim GloVe model at {gensim_model_abspath}')
            os.makedirs(gensim_model_abspath)
            glove.save(os.path.join(gensim_model_abspath, 'glove'))
        print(f'Loading GloVe vector gensim model from {gensim_model_abspath}/glove.')
        os.remove(tmp_file)
    else:
        print(f'GloVe gensim KeyedVectors model exists! Loading model from {gensim_model_abspath}/glove.')
        glove = gensim.models.KeyedVectors.load(os.path.join(gensim_model_abspath, 'glove'))
    return glove


if __name__ == '__main__':
    GLOVE_PATH = os.path.expanduser('~/asr_simulator_data/glove')
    if not Path(GLOVE_PATH).is_dir():
        print(f'Creating directory for glove model at {GLOVE_PATH}')
        os.makedirs(GLOVE_PATH)

    glove = load_glove_model(GLOVE_PATH)
