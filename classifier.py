import pandas as pd
from data_pipeline import get_session_sentences, \
                            get_concatenated_session_contents, \
                            preprocess_docs,\
                            get_original_and_translated_sentences
from argparser import get_argparser
from topic_modelling import get_topic_feature_gender_vectors, get_topic_feature_country_vectors, get_topic_feature_age_vectors
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from doc_representation import get_bow_vectors
from sklearn.decomposition import PCA

def cv_evaluate_topics_in_classification(full_session_data, get_features, model_id, lang, kernel, C, gamma, num_topics, splits):
    print('Evaluating', num_topics, 'in classification')
    topic_features, labels = get_features(full_session_data, num_topics, model_id, lang)
    classifier = SVC(kernel=kernel, C=C, gamma=gamma)
    scores = cross_val_score(classifier, topic_features, labels, cv=splits, n_jobs=-1)
    del classifier, topic_features, labels
    return scores.mean()

def cv_evaluate_bow_in_classification(bow_features, labels, feature_size, kernel, C, gamma, splits):
    pca = PCA(n_components=feature_size)
    bow_features = pca.fit_transform(bow_features)    
    classifier = SVC(kernel=kernel, C=C, gamma=gamma)
    scores = cross_val_score(classifier, bow_features, labels, cv=splits, n_jobs=-1)
    del classifier, labels
    return scores.mean()

def evaluate_svm(X_train, y_train, X_test, y_test, **kwargs):
    classifier = SVC(**kwargs)
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    f1_score = f1_score(y_test, preds)
    accuracy = classifier.score(X_test, y_test)
    return accuracy, f1_score

def optimise_num_of_topics(X, get_features, model_id, lang, splits, kernels=['rbf', 'linear', 'poly'],\
                            topic_nums=[50, 100, 200, 500, 1000],\
                            Cs=[0.5, 1, 2, 5, 10, 100], gammas=['auto', 'scale', 0.1, 0.5, 1, 5]):
    best_score = 0
    best_hyperparams = None
    for kernel in kernels:
        for n_topics in topic_nums:
            for c in Cs:
                for gamma in gammas:
                    print('---------------')
                    mean_score = cv_evaluate_topics_in_classification(X, get_features, model_id, lang, kernel, c, gamma, n_topics, splits)
                    print('mean score', mean_score, 'with', kernel, 'and', n_topics, 'topics and C =', c, 'and gamma =', gamma)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_hyperparams = (kernel, n_topics, c, gamma)
                    print('best score', best_score, 'with ', best_hyperparams)
    print('Best score of', best_score, 'obtained with', best_hyperparams)

def optimise_svm_hyperparams(X, y, splits, kernels=['rbf', 'linear', 'poly'], feature_size=[50, 100, 200, 500, 1000],\
                            Cs=[0.5, 1, 2, 5, 10, 100], gammas=['auto', 'scale', 0.1, 0.5, 1, 5]):
    best_score = 0
    best_hyperparams = None
    for kernel in kernels:
        for n in feature_size:
            for c in Cs:
                for gamma in gammas:
                    print('---------------')
                    mean_score = cv_evaluate_bow_in_classification(X, y, n, kernel, c, gamma, splits)
                    print('mean score', mean_score, 'with', kernel, 'and feature size is', n, 'and C =', c, 'and gamma =', gamma)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_hyperparams = (kernel, c, gamma)
                    print('best score', best_score, 'with ', best_hyperparams)
    print('Best score of', best_score, 'obtained with', best_hyperparams)

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    english_session_sentences = get_session_sentences(args.speaker_info, args.english_file)
    german_session_sentences = get_session_sentences(args.speaker_info, args.german_file)
    english_original, translated_to_english, german_original, translated_to_german = get_original_and_translated_sentences(english_session_sentences, german_session_sentences)
    print('Preprocessing session contents')
    full_english_session_data = get_concatenated_session_contents(english_session_sentences)
    full_german_session_data = get_concatenated_session_contents(german_session_sentences)
    original_english_session_data = get_concatenated_session_contents(english_original)
    original_german_session_data = get_concatenated_session_contents(german_original)
    translated_to_english_session_data = get_concatenated_session_contents(translated_to_english)
    translated_to_german_session_data = get_concatenated_session_contents(translated_to_german)
    print('Dividing datasets')
    all_english_train, all_english_test = train_test_split(full_english_session_data, test_size=0.2, random_state=0)
    all_german_train, all_german_test = train_test_split(full_german_session_data, test_size=0.2, random_state=0)
    orig_english_train, orig_english_test = train_test_split(original_english_session_data, test_size=0.2, random_state=0)
    orig_german_train, orig_german_test = train_test_split(original_german_session_data, test_size=0.2, random_state=0)
    trans_english_train, trans_english_test = train_test_split(translated_to_english_session_data, test_size=0.2, random_state=0)
    trans_german_train, trans_german_test = train_test_split(translated_to_german_session_data, test_size=0.2, random_state=0)
    print('Done')
    ######## USING TOPICS AS FEATURES #######
    # Using English data
    #optimise_num_of_topics(all_english_train, get_topic_feature_age_vectors, 'all_english', 'en', splits=3)
    optimise_num_of_topics(all_english_train, get_topic_feature_gender_vectors, 'all_english', 'en', splits=3)
    # Using German data
    #optimise_num_of_topics(all_german_train, get_topic_feature_age_vectors, 'all_german', 'de_core_news_sm', splits=3)
    optimise_num_of_topics(all_german_train, get_topic_feature_gender_vectors, 'all_german', 'de_core_news_sm', splits=3)
    # Using original language data
    # English
    print('Found', len(original_english_session_data), 'session contents with English as original language')
    optimise_num_of_topics(orig_english_train, get_topic_feature_gender_vectors, 'orig_english', 'en', splits=3)
    # German
    print('Found', len(original_german_session_data), 'session contents with German as original language')
    optimise_num_of_topics(orig_german_train, get_topic_feature_gender_vectors, 'orig_german', 'de_core_news_sm', splits=3)
    # Using translated language data
    # English
    print('Found', len(translated_to_english_session_data), 'session contents translated to English')
    optimise_num_of_topics(trans_english_train, get_topic_feature_gender_vectors, 'trans_english', 'en', splits=3)
    # German
    print('Found', len(translated_to_german_session_data), 'session contents translated to German')
    optimise_num_of_topics(trans_german_train, get_topic_feature_gender_vectors, 'trans_german', 'de_core_news_sm', splits=3)
    ######## USING BOW FEATURES ###########
    # all English
    train_all_english_bows, all_english_vectoriser = get_bow_vectors(all_english_train['content'].values)
    train_all_english_labels = all_english_train['gender'].values
    optimise_svm_hyperparams(train_all_english_bows, train_all_english_labels, 3)
    # all German
    train_all_german_bows, all_german_vectoriser = get_bow_vectors(all_german_train['content'].values)
    train_all_german_labels = all_german_train['gender'].values
    optimise_svm_hyperparams(train_all_german_bows, train_all_german_labels, 3)
    # orig English
    train_orig_english_bows, orig_english_vectoriser = get_bow_vectors(orig_english_train['content'].values)
    train_orig_english_labels = orig_english_train['gender'].values
    optimise_svm_hyperparams(train_orig_english_bows, train_orig_english_labels, 3)
    # orig German
    train_orig_german_bows, orig_german_vectoriser = get_bow_vectors(orig_german_train['content'].values)
    train_orig_german_labels = orig_german_train['gender'].values
    optimise_svm_hyperparams(train_orig_german_bows, train_orig_german_labels, 3)
    # trans English
    train_trans_english_bows, trans_english_vectoriser = get_bow_vectors(trans_english_train['content'].values)
    train_trans_english_labels = trans_english_train['gender'].values
    optimise_svm_hyperparams(train_trans_english_bows, train_trans_english_labels, 3)
    # trans German
    train_trans_german_bows, trans_german_vectoriser = get_bow_vectors(trans_german_train['content'].values)
    train_trans_german_labels = trans_german_train['gender'].values
    optimise_svm_hyperparams(train_trans_german_bows, train_trans_german_labels, 3)
