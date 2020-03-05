import re
import utils
from collections import defaultdict
import pandas as pd
import spacy
from tqdm import tqdm
import pickle
import os
from spacy.lang.en.stop_words import STOP_WORDS
import utils

def build_speaker_info(raw_speaker_info):
    attributes = ['COUNT', 'EUROID', 'NAME', 'LANGUAGE', 'GENDER', 'DATE_OF_BIRTH', 'SESSION_DATE', 'AGE']
    numerical = ['AGE']
    speaker_dict = defaultdict(list)
    for att in attributes:
        for info in raw_speaker_info:
            value = re.search(rf" {att}=\"(.*?)\"", info).group(1)
            if att in numerical:
                value = int(value)
            speaker_dict[att.lower()].append(value)
    speakers = pd.DataFrame.from_dict(speaker_dict)
    return speakers

def get_session_sentences(speaker_info_file, sentence_file):
    raw_speaker_info = utils.read_file_lines(speaker_info_file)
    speaker_info = build_speaker_info(raw_speaker_info)
    sentences = utils.read_file_lines(sentence_file)
    sentence_df = pd.DataFrame(sentences, columns=['sentence'])
    data = pd.merge(speaker_info, sentence_df, left_index=True, right_index=True)
    return data

def get_concatenated_session_contents(data):
    grouped_by_session_and_id = data.groupby(['session_date', 'euroid'])
    session_dict = defaultdict(list)
    i = 0
    for name, group in grouped_by_session_and_id:
        session_content = ' '.join(group['sentence'].values)
        session_dict['content'].append(session_content)
        cols_to_skip = ['sentence', 'count', 'language', 'name']
        for col in group.columns:
            if col in cols_to_skip:
                continue
            session_dict[col].append(group.iloc[0][col])
        session_dict['id'].append(i)
        i += 1
    session_contents = pd.DataFrame.from_dict(session_dict)
    return session_contents

def preprocess_docs(docs, lang, model_id, cache_path='cache/preprocessed_docs'):
    cache_path += '_' + model_id
    if os.path.exists(cache_path):
        print('Loading preprocessed docs from', cache_path)
        preprocessed_docs = pickle.load(open(cache_path, 'rb'))
        return preprocessed_docs
    nlp = spacy.load(lang)
    processed_docs = []
    custom_stopwords = utils.read_file_lines('stopwords.txt')
    for doc in tqdm(nlp.pipe(docs, n_threads=4, batch_size=1000), total=len(docs), desc="Preprocessing docs", mininterval=0.2):
        ents = doc.ents  # Named entities.
        doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        doc = [token for token in doc if token not in STOP_WORDS and token not in custom_stopwords]
        doc.extend([str(entity) for entity in ents if len(entity) > 1])
        processed_docs.append(doc)
    print('Saving preprocessed docs to', cache_path)
    pickle.dump(processed_docs, open(cache_path, 'wb'))
    return processed_docs

def get_original_and_translated_sentences(english_session_sentences, german_session_sentences):
    english_original_sentences = english_session_sentences[english_session_sentences['language'] == 'EN'].copy()
    german_original_sentences = german_session_sentences[german_session_sentences['language'] == 'DE'].copy()
    translated_to_english_sentences = english_session_sentences[english_session_sentences['language'] == 'DE'].copy()
    translated_to_german_sentences = german_session_sentences[german_session_sentences['language'] == 'EN'].copy()
    return english_original_sentences, translated_to_english_sentences, german_original_sentences, translated_to_german_sentences