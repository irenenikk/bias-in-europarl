# This file is largely based on this tutorial by Gensim:
# https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb#Explore-author-topic-representation

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import AuthorTopicModel, atmodel, LdaModel
from gensim.models.coherencemodel import CoherenceModel
from data_pipeline import preprocess_docs, get_age_bin, get_age_bin2

def build_author2doc(data):
    print('Building author2doc dict')
    author2doc = defaultdict(list)
    for i in range(len(data)):
        euroid = data['euroid'].iloc[i]
        author2doc[euroid].append(i)
    return author2doc

def build_dictionary(docs):
    print('Building dictionary')
    dictionary = Dictionary(docs)
    max_freq = 0.5
    min_wordcount = 20
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
    _ = dictionary[0]  # This sort of "initializes" dictionary.id2token.
    return dictionary

def build_bow_corpus(docs, dictionary):
    print('Building corpus')
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return corpus

def get_author_topic_model(full_session_data, num_topics, model_id, lang, model_path='cache/model.atmodel', passes=5, iterations=20, verbose=False):
    model_path = model_path + '_' + model_id + '_' + str(num_topics)
    if os.path.exists(model_path):
        print('Loading an author topic model from', model_path)
        model = AuthorTopicModel.load(model_path)
        return model
    print('Training a new author topic model with id', model_id)
    raw_docs = full_session_data['content'].values
    docs = preprocess_docs(raw_docs, lang, model_id)
    author2doc = build_author2doc(full_session_data)
    dictionary = build_dictionary(docs)
    corpus = build_bow_corpus(docs, dictionary)
    if verbose:
        print(f'Number of authors: {len(author2doc)}')
        print(f'Number of unique tokens: {len(dictionary)}')
        print(f'Number of documents: {len(corpus)}')
    model = AuthorTopicModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, \
                            author2doc=author2doc, chunksize=3000, passes=passes, eval_every=5, \
                            iterations=iterations, random_state=1)
    model.save(model_path)
    print('Saving author topic model to', model_path)
    return model

def get_lda_topic_model(full_session_data, num_topics, model_id, lang,\
                        model_path='cache/model.ldamodel', passes=5, iterations=20, verbose=False):
    model_path = model_path + '_' + model_id + '_' + str(num_topics)
    if os.path.exists(model_path):
        print('Loading an LDA model from', model_path)
        model = LdaModel.load(model_path)
        return model
    print('Training a new LDA topic model with id', model_id)
    raw_docs = full_session_data['content'].values
    docs = preprocess_docs(raw_docs, lang, model_id)
    dictionary = build_dictionary(docs)
    corpus = build_bow_corpus(docs, dictionary)
    if verbose:
        print(f'Number of unique tokens: {len(dictionary)}')
        print(f'Number of documents: {len(corpus)}')
    model = LdaModel(corpus=corpus, num_topics=num_topics, alpha='auto', id2word=dictionary,\
                    eta='auto', chunksize=2000, passes=passes, eval_every=5, \
                    iterations=iterations, random_state=1)
    model.save(model_path)
    print('Saving LDA topic model to', model_path)
    return model

def calculate_per_word_bound(model):
    corpus_words = sum(cnt for document in model.corpus for _, cnt in document)
    perwordbound = model.bound(model.corpus, author2doc=model.author2doc, \
                            doc2author=model.doc2author) / corpus_words
    return perwordbound

def calculate_topic_coherence(model):
    cm = CoherenceModel(model=model, corpus=model.corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    return coherence

def find_best_amount_of_topics_on_increasing_metric(full_session_data, metric, model_id, lang, options=[5, 10, 20, 50, 100, 200, 300, 500, 1000]):
    best_number_of_topics = 0
    best = float('-inf')
    for n in options:
        print('Evaluating topic coherence with', n, 'topics')
        model = get_author_topic_model(full_session_data, n, model_id, lang)
        result = metric(model)
        print('Result', result)
        if result > best:
            best = result
            best_number_of_topics = n
    return best_number_of_topics

def find_best_amount_of_topics_on_decreasing_metric(full_session_data, metric, model_id, lang, options=[200, 300, 500, 1000, 2000, 5000]):
    best_number_of_topics = 0
    best = float('inf')
    for n in options:
        print('Evaluating per word bound with', n, 'topics')
        model = get_author_topic_model(full_session_data, n, model_id, lang)
        result = metric(model)
        if result < best:
            best = result
            best_number_of_topics = n
    return best_number_of_topics

def build_topic_feature_gender_data(full_session_data, model):
    euroids = full_session_data['euroid'].unique()
    topic_dists = np.zeros((len(euroids), model.num_topics))
    genders = np.zeros(len(euroids))
    for i, euroid in enumerate(euroids):
        try:
            topics = model.get_author_topics(str(euroid), minimum_probability=0)
            topic_probs = [topic[1] for topic in topics]
            topic_dists[i] = topic_probs
            gender = full_session_data[full_session_data['euroid'] == euroid]['gender'].mode()[0]
            if gender == 'MALE':
                genders[i] = 1
        except:
            # mep not found in model
            pass
    return topic_dists, genders

def build_topic_feature_country_data(full_session_data, model):
    mep_details = pd.read_csv('europarl/MEPs.details.txt', sep='\t')
    euroids = full_session_data['euroid'].unique()
    topic_dists = np.zeros((len(euroids), model.num_topics))
    countries = []
    for i, euroid in enumerate(euroids):
        try:
            topics = model.get_author_topics(str(euroid), minimum_probability=0)
            topic_probs = [topic[1] for topic in topics]
            topic_dists[i] = topic_probs
            country = mep_details[mep_details['MEP id'] == int(euroid)]['Country'].values[0]
            countries.append(country)
        except:
            # mep not found in model
            pass
    return topic_dists, countries

def build_topic_feature_age_data(full_session_data, model):
    topic_dists = np.zeros((len(full_session_data), model.num_topics))
    ages = np.zeros(len(full_session_data))
    for i in range(len(full_session_data)):
        content = full_session_data.iloc[i]['content']
        content_bow = model.id2word.doc2bow(content.split(" "))
        topics = model.get_document_topics(content_bow, minimum_probability=0)
        topic_probs = [topic[1] for topic in topics]
        topic_dists[i] = topic_probs
        age = get_age_bin(full_session_data.iloc[i]['age'])
        ages[i] = age
    return topic_dists, ages

def get_topic_feature_gender_vectors(full_session_data, num_topics, model_id, lang):
    ''' Build topic feature vectors for session data associated with each MEP '''
    model = get_author_topic_model(full_session_data, num_topics, model_id=model_id, lang=lang)
    topic_dists, genders = build_topic_feature_gender_data(full_session_data, model)
    print('Gender distribution')
    print(pd.DataFrame(genders)[0].value_counts(normalize=True))
    return topic_dists, genders

def get_topic_feature_country_vectors(full_session_data, num_topics, model_id, lang):
    ''' Build topic feature vectors for session data associated with each MEP '''
    model = get_author_topic_model(full_session_data, num_topics, model_id=model_id, lang=lang)
    topic_dists, countries = build_topic_feature_country_data(full_session_data, model)
    print('Country distribution')
    print(pd.DataFrame(countries)[0].value_counts(normalize=True))
    return topic_dists, countries

def get_topic_feature_age_vectors(full_session_data, num_topics, model_id, lang):
    model = get_lda_topic_model(full_session_data, num_topics, model_id=model_id, lang=lang)
    topic_dists, ages = build_topic_feature_age_data(full_session_data, model)
    print('Age distribution')
    print(pd.DataFrame(ages)[0].value_counts(normalize=True))
    return topic_dists, ages