from sklearn.feature_extraction.text import CountVectorizer


# Code reused from previous practical https://github.com/irenenikk/sentiment-analysis-comparison
def get_bow_vectors(docs, min_count=20, max_frac=.5, vectorizer=None, frequency=True):
    print('Getting', len(docs), 'BOW vectors')
    if vectorizer is None:
        vectorizer = CountVectorizer(min_df=min_count, max_df=max_frac, lowercase=True, ngram_range=(1, 1))
        vectorizer.fit(docs)
    bows = vectorizer.transform(docs).toarray()    
    print('BOW vector shape', bows.shape)
    if frequency:
        return bows, vectorizer
    # if frequency is false only look at presence of feature
    bows[bows > 0] = 1
    return bows, vectorizer
