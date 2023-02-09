import string
import emoji  # Use version emoji==1.6.1, otherwise it won't have UNICODE_EMOJI


main_special_characters = string.punctuation + string.digits + string.whitespace
other_special_characters = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)
emoji = list(emoji.UNICODE_EMOJI["en"].keys())

special_characters_default = set(main_special_characters + other_special_characters)
special_characters_default.update(emoji)


parameters_filtering_default = {
    "cond_uniform_whitespace": True,
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": False,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": False,
    "length_word_max_cutoff": 50,
    "cond_check_number_words": True,
    "tokenization": False,
    "strip_characters": special_characters_default,
    "number_words_min_cutoff": 10,
    "number_words_max_cutoff": 100000,
    "cond_check_character_repetition_removal": True,
    "character_repetition_length": 10,
    "character_repetition_max_cutoff": 0.2,
    "cond_check_word_repetition_removal": True,
    "word_repetition_length": 5,
    "word_repetition_max_cutoff": 0.3,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.4,
    "cond_words_augmentation": False,
    "words_augmentation_group_sizes": [],
    "words_augmentation_join_char": "",
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.1,
    "cond_check_flagged_words": True,
    "flagged_words_max_cutoff": 0.1,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.70,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 10000,
}

parameters_filtering_en = {
    "cond_uniform_whitespace": True,
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": True,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 25,
    "cond_check_number_words": True,
    "tokenization": False,
    "strip_characters": special_characters_default,
    "number_words_min_cutoff": 20,
    "number_words_max_cutoff": 100000,
    "cond_check_character_repetition_removal": True,
    "character_repetition_length": 10,
    "character_repetition_max_cutoff": 0.106,
    "cond_check_word_repetition_removal": True,
    "word_repetition_length": 5,
    "word_repetition_max_cutoff": 0.19,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.4,
    "cond_words_augmentation": False,
    "words_augmentation_group_sizes": [],
    "words_augmentation_join_char": "",
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.3,
    "cond_check_flagged_words": True,
    "flagged_words_max_cutoff": 0.01,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.80,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 1500,
}

parameters_filtering_fr = {
    "cond_uniform_whitespace": True,
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": True,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 45,
    "cond_check_number_words": True,
    "tokenization": False,
    "strip_characters": special_characters_default,
    "number_words_min_cutoff": 13,
    "number_words_max_cutoff": 100000,
    "cond_check_character_repetition_removal": True,
    "character_repetition_length": 10,
    "character_repetition_max_cutoff": 0.14,
    "cond_check_word_repetition_removal": True,
    "word_repetition_length": 5,
    "word_repetition_max_cutoff": 0.13,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.34,
    "cond_words_augmentation": False,
    "words_augmentation_group_sizes": [],
    "words_augmentation_join_char": "",
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.27,
    "cond_check_flagged_words": True,
    "flagged_words_max_cutoff": 0.008,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.8,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 1770,
}

# TO-DO: Adapt these parametets to Turkish. See this: https://github.com/bigscience-workshop/data-preparation/tree/main/preprocessing/training/01b_oscar_cleaning_and_filtering/visualization
# You can use the API: https://huggingface.co/spaces/huggingface/text-data-filtering. Play and try to observe for 50 samples!
parameters_filtering_tr = {
    "cond_uniform_whitespace": True,
    "cond_replace_unicode_punctuation": False,
    "cond_remove_words_with_incorrect_substrings": True,
    "incorrect_word_substrings": ["http", "www", ".com", "href", "//"],
    "cond_remove_long_words": True,
    "length_word_max_cutoff": 25,
    "cond_check_number_words": True,
    "tokenization": False,
    "strip_characters": special_characters_default,
    "number_words_min_cutoff": 20,
    "number_words_max_cutoff": 100000,
    "cond_check_character_repetition_removal": True,
    "character_repetition_length": 10,
    "character_repetition_max_cutoff": 0.106,
    "cond_check_word_repetition_removal": True,
    "word_repetition_length": 5,
    "word_repetition_max_cutoff": 0.19,
    "cond_check_special_characters": True,
    "special_characters": special_characters_default,
    "special_characters_max_cutoff": 0.4,
    "cond_words_augmentation": False,
    "words_augmentation_group_sizes": [],
    "words_augmentation_join_char": "",
    "cond_check_stopwords": True,
    "stopwords_min_cutoff": 0.3,
    "cond_check_flagged_words": True,
    "flagged_words_max_cutoff": 0.01,
    "cond_check_lang_id": True,
    "lang_id_min_cutoff": 0.80,
    "cond_check_perplexity": True,
    "perplexity_max_cutoff": 1500,
}

parameters_filtering = {
    "default": parameters_filtering_default,
    "en": parameters_filtering_en,
    "tr": parameters_filtering_tr,
    "fr": parameters_filtering_fr,
}
