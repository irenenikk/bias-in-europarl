# This file is largely based on this tutorial by Gensim:
# https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb#Explore-author-topic-representation

from argparser import get_argparser
from data_pipeline import get_session_sentences, \
                            get_concatenated_session_contents, \
                            preprocess_docs
import pandas as pd
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import AuthorTopicModel, atmodel
import os

def build_author2doc(data):
    print('Building author2doc dict')
    author2doc = defaultdict(list)
    for euroid, doc_id in data[['euroid', 'id']].values:
        author2doc[euroid].append(doc_id)
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

def get_author_topic_model(full_session_data, model_path='cache/model.atmodel', num_topics=20, verbose=False):
    model_path = model_path + '_' + str(num_topics)
    if os.path.exists(model_path):
        print('Loading a author topic model from', model_path)
        model = AuthorTopicModel.load(model_path)
        return model
    print('Training a new author topic model')
    raw_docs = full_session_data['content'].values
    docs = preprocess_docs(raw_docs, 'en')
    author2doc = build_author2doc(full_session_data)
    dictionary = build_dictionary(docs)
    corpus = build_bow_corpus(docs, dictionary)
    if verbose:
        print(f'Number of authors: {len(author2doc)}')
        print(f'Number of unique tokens: {len(dictionary)}')
        print(f'Number of documents: {len(corpus)}')
    model = AuthorTopicModel(corpus=corpus, num_topics=num_topics, id2word=dictionary.id2token, \
                    author2doc=author2doc, chunksize=2000, passes=1, eval_every=5, \
                    iterations=1, random_state=1)
    model.save(model_path)
    print('Saving author topic model to', model_path)
    return model

def calculate_per_word_bound(model):
    corpus_words = sum(cnt for document in model.corpus for _, cnt in document)
    perwordbound = model.bound(model.corpus, author2doc=model.author2doc, \
                            doc2author=model.doc2author) / corpus_words
    return perwordbound

def find_best_amount_of_topics(full_session_data, options=[200, 300, 500, 1000, 2000, 5000]):
    best_number_of_topics = 0
    best_perwordbound = 0
    for n in options:
        print('Evaluating per word bound with', n, 'topics')
        model = get_author_topic_model(full_session_data, num_topics=n)
        perwordbound = calculate_per_word_bound(model)
        if perwordbound < best_perwordbound:
            best_perwordbound = perwordbound
            best_number_of_topics = n
    return best_number_of_topics

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    mep_details = pd.read_csv(args.mep_details, sep='\t')
    session_sentences = get_session_sentences(args.speaker_info, args.target_lang)
    full_session_data = get_concatenated_session_contents(session_sentences)
    best_num_of_topics = find_best_amount_of_topics(full_session_data)
    print(best_num_of_topics)