asr-error-simulator
==============================

Word substitution error simulator to produce ASR-plausible errors in a corpus of text. The simulator considers both semantic information (via GloVe word embeddings) and acoustic/phonetic information (by computing the phonological edit distance).

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
