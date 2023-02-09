## Data Filtering and Data Deduplication of the BigScience Corpus

This is the data filtering code used to clean the Oscar subset of the ROOTS dataset.

The supported languages are defined in the file [languages_id.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/languages_id.py).
We should add Turkish (`tr`) as a new field to `languages_id.py`. 

### Filtering

#### 0. Understand the filtering pipeline

Please take a look at the pdf [explanation filtering pipeline](https://drive.google.com/file/d/1cCJ8sWE88TRLDAa3eHLmXO4JlkR2QzLY/view?usp=sharing) for an explanation of the filtering pipeline. Try to understand each step and take notes for important points. 

#### 1. Define the lists of stop words and flagged words, and check how the normalization of texts are done

You might want to add/redefine the lists of stop words (closed class words) and flagged words for `Turkish` which are must for robustness or ethical reasons in the files [stopwords.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/stopwords.py) and [flagged_words.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/flagged_words.py).

You can also check how the normalization of texts are done in the files [normalization.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/normalization.py). You can add more techniques from [huggingface](https://huggingface.co/docs/tokenizers/api/normalizers).

#### 2. Download everything you need

To run the filtering code, it is necessary to download the dataset on which the filtering will take place, but also the necessary models, which are the Fasttext model for language identification (download [here](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin)) and the Sentencepiece and KenLM models for tokenization and calculation of perplexity scores (download with the file [download_sentencepiece_kenlm_models.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/download_sentencepiece_kenlm_models.py)).

#### 3. Choose the filtering parameters

The filtering parameters for each language are to be specified in the file [parameters_filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/parameters_filtering.py). It is **strongly recommended** to look at the data and use the visualization code in the directory [visualization](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/01b_oscar_cleaning_and_filtering/visualization) to choose these parameters.

#### 4. Run the filtering

Run the filtering with the file [main_filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/main_filtering.py), specifying the dataset used and the links to the downloaded models. The different filters are coded in the file [filtering.py](https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01b_oscar_cleaning_and_filtering/filtering.py).

#### 5. Do the deduplication
**Note:** Please see Emre Can before that point.
Do the deduplication, which is detailed in the sub folder [deduplicate](https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/01b_oscar_cleaning_and_filtering/deduplicate).
