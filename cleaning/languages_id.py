import pandas as pd


langs_id = [
    {
        "lang": "Turkish",
        "dataset_id": "tr",
        "stopwords_id": "tr",
        "flagged_words_id": "tr",
        "fasttext_id": "tr",
        "sentencepiece_id": "tr",
        "kenlm_id": "tr",
    },
    {
        "lang": "Arabic",
        "dataset_id": "ar",
        "stopwords_id": "ar",
        "flagged_words_id": "ar",
        "fasttext_id": "ar",
        "sentencepiece_id": "ar",
        "kenlm_id": "ar",
    },
    {
        "lang": "Bengali",
        "dataset_id": "bn",
        "stopwords_id": "bn",
        "flagged_words_id": "bn",
        "fasttext_id": "bn",
        "sentencepiece_id": "bn",
        "kenlm_id": "bn",
    },
    {
        "lang": "Catalan",
        "dataset_id": "ca",
        "stopwords_id": "ca",
        "flagged_words_id": "ca",
        "fasttext_id": "ca",
        "sentencepiece_id": "ca",
        "kenlm_id": "ca",
    },
    {
        "lang": "English",
        "dataset_id": "en",
        "stopwords_id": "en",
        "flagged_words_id": "en",
        "fasttext_id": "en",
        "sentencepiece_id": "en",
        "kenlm_id": "en",
    },
    {
        "lang": "Spanish",
        "dataset_id": "es",
        "stopwords_id": "es",
        "flagged_words_id": "es",
        "fasttext_id": "es",
        "sentencepiece_id": "es",
        "kenlm_id": "es",
    },
    {
        "lang": "Basque",
        "dataset_id": "eu",
        "stopwords_id": "eu",
        "flagged_words_id": "eu",
        "fasttext_id": "eu",
        "sentencepiece_id": "eu",
        "kenlm_id": "eu",
    },
    {
        "lang": "French",
        "dataset_id": "fr",
        "stopwords_id": "fr",
        "flagged_words_id": "fr",
        "fasttext_id": "fr",
        "sentencepiece_id": "fr",
        "kenlm_id": "fr",
    },
    {
        "lang": "Hindi",
        "dataset_id": "hi",
        "stopwords_id": "hi",
        "flagged_words_id": "hi",
        "fasttext_id": "hi",
        "sentencepiece_id": "hi",
        "kenlm_id": "hi",
    },
    {
        "lang": "Indonesian",
        "dataset_id": "id",
        "stopwords_id": "id",
        "flagged_words_id": "id",
        "fasttext_id": "id",
        "sentencepiece_id": "id",
        "kenlm_id": "id",
    },
    {
        "lang": "Portuguese",
        "dataset_id": "pt",
        "stopwords_id": "pt",
        "flagged_words_id": "pt",
        "fasttext_id": "pt",
        "sentencepiece_id": "pt",
        "kenlm_id": "pt",
    },
    {
        "lang": "Urdu",
        "dataset_id": "ur",
        "stopwords_id": "ur",
        "flagged_words_id": "ur",
        "fasttext_id": "ur",
        "sentencepiece_id": "ur",
        "kenlm_id": "ur",
    },
    {
        "lang": "Vietnamese",
        "dataset_id": "vi",
        "stopwords_id": "vi",
        "flagged_words_id": "vi",
        "fasttext_id": "vi",
        "sentencepiece_id": "vi",
        "kenlm_id": "vi",
    },
    {
        "lang": "Chinese",
        "dataset_id": "zh",
        "stopwords_id": "zh",
        "flagged_words_id": "zh",
        "fasttext_id": "zh",
        "sentencepiece_id": "zh",
        "kenlm_id": "zh",
    },
]
langs_id = pd.DataFrame(langs_id)
