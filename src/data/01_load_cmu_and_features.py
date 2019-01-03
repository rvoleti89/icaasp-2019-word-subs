import csv
from corpustools.corpus import classes
from corpustools.corpus import io
from pathlib import Path
import os
import wget
import pickle


def read_cmu(f):
    cmu_dict = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in tsv_reader:
            spelling = row[0]
            transcription = row[1].split()
            # transcription = classes.lexicon.Transcription(row[1].split())

            try:
                word = classes.lexicon.Word(symbol=spelling, transcription=transcription)
            except:
                continue
            # setattr(word, 'transcription', transcription)
            cmu_dict[spelling] = word
    return cmu_dict


def download_cmu(path):
    url = 'https://raw.githubusercontent.com/' \
          'kelvinguu/simple-speech-recognition/master/lib/models/cmudict.0.7a_SPHINX_40'

    wget.download(url, os.path.join(path, 'cmudict.0.7a_SPHINX_40'))


if __name__ == '__main__':
    if not Path('../models/dict.pkl').is_file():
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
            download_cmu(data_path)

        cmu_dict = read_cmu(os.path.join(data_path, 'cmudict.0.7a_SPHINX_40'))

        # Save pickle file for cmu_dict and arpabet2hayes for easier loading next time
        with open('../models/dict.pkl', 'wb') as f:
            pickle.dump([cmu_dict, arpabet2hayes], f)
    else:
        with open('../models/dict.pkl', 'rb') as f:
            cmu_dict, arpabet2hayes = pickle.load(f)

    num_words = len(cmu_dict)
    print(f'Loaded CMU Pronouncing Dictionary with ARPABET transcriptions for {num_words} English words '
          f'and feature matrix by Hayes.')
