from setuptools import setup

setup(name='asr_error_simulator',
      version='0.1',
      description='Corrupts input text with a given Word Error Rate (WER) with ASR-plausible word substitution errors. '
                  'Makes use of GloVe vectors (modeling semantics) and the phonological edit distance (modeling '
                  'acoustics).',
      # url='',
      author='Rohit Voleti',
      author_email='rnvoleti@asu.edu',
      license='BSD',
      packages=['src', 'src.data', 'src.tools', 'src.models'],
      install_requires=[
          'numpy',
          'pandas',
          'gensim',
          'nltk',
          'wget',
          'progressbar2',
          'corpustools @ https://github.com/PhonologicalCorpusTools/CorpusTools/archive/v1.3.0.zip'
      ],
      scripts=['src/corrupt_text_file.py'],
      python_requires='>=3.6.6, <=3.7',
      zip_safe=False)
