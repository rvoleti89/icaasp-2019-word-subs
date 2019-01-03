asr-error-simulator
==============================

Word substitution error simulator to produce ASR-plausible errors in a corpus of text. 
The simulator considers both semantic information (via [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/)) and acoustic/phonetic information (by computing the [phonological edit distance](https://corpustools.readthedocs.io/en/latest/string_similarity.html)).

**Note**: Currently this works using the phonological corpus tools package v1.3.0, the latest 1.4.0 seems to have an issue. Will update if this changes.

The method is described in our [paper](https://arxiv.org/abs/1811.07021), currently under review for ICASSP-2019.

Project Organization
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── bin
    ├── config
    ├── data
    │   ├── external
    │   ├── interim
    │   ├── processed
    │   └── raw
    ├── docs
    ├── notebooks
    ├── reports
    │   └── figures
    └── src
        ├── data
        ├── external
        ├── models
        ├── tools
        └── visualization

### Installation
##### Option 1: Install *pip* package
This repository can be pip installed with the following command:
```bash
pip install +git...
```
##### Option 2: Clone repository
Clone the repository, cd into the directory, and install with *pip*:
```bash
git clone ...
cd asr_error_simulator
pip install -e .
```

### Usage:
Installing the package includes adding a python script titled *corrupt_text_file.py* to the `PATH` in your environment.

*corrupt_text_file.py* supports the following command line arguments:
* **Required**:
  * `-f`: Path to a plaintext file which contains original text to be corrupted
* **Optional**:
  * `-e`: Specified word error rate (WER), a value between 0.0 and 1.0 which determines the percentage of words in the provided file to be replaced with ASR-plausible confusion words. If not specified, the default value is $0.30$, Though this argument is optional, it is recommended to specify a WER when running the script.
  * `-o`: Path to directory where processed output file is to be stored. If not specified, output location defaults to the same directory as the specified text file.
  * `--redo`: A boolean flag which is set to `False` by default. Can be set to `True` if the user would like to recompute all phonological edit distances and re-download the models after the first run. Not recommended.
  * `--glove`: Can optionally specify a directory which contains the GloVe vectors .txt file from Stanford NLP if already downloaded. File will be downloaded to asr_error_simulator data directory in home folder by default otherwise.
  
##### Usage example:

```bash
corrupt_text_file.py -f ~/Desktop/test.txt -e 0.347
```