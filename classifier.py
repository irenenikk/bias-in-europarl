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

def evaluate_topics_in_classification(full_session_data, get_features, model_id, lang, kernel, C, gamma, num_topics, splits, reduced_dim=100):
    print('Evaluating', num_topics, 'in classification')
    topic_features, labels = get_features(full_session_data, num_topics, model_id, lang)
    if topic_features.shape[1] > reduced_dim:
        print('Reducing num of topics from', num_topics, 'to', reduced_dim)
        pca = PCA(n_components=reduced_dim)
        topic_features = pca.fit_transform(topic_features)
    classifier = SVC(kernel=kernel, C=C, gamma=gamma)
    scores = cross_val_score(classifier, topic_features, labels, cv=splits, n_jobs=-1)
    del classifier, topic_features, labels
    return scores.mean()

def optimise_num_of_topics(X, get_features, model_id, lang, splits=5, kernels=['rbf', 'linear', 'poly'],\
                            topic_nums=[10, 50, 100, 200, 500, 1000],\
                            Cs=[1], gammas=['auto', 'scale']):
    best_score = 0
    best_hyperparams = None
    for kernel in kernels:
        for n_topics in topic_nums:
            for c in Cs:
                for gamma in gammas:
                    print('---------------')
                    mean_score = evaluate_topics_in_classification(X, get_features, model_id, lang, kernel, c, gamma, n_topics, splits, 200)
                    print('mean score', mean_score, 'with', kernel, 'and', n_topics, 'topics and C =', c, 'and gamma =', gamma)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_hyperparams = (kernel, n_topics, c, gamma)
                    print('best score', best_score, 'with ', best_hyperparams)
    print('Best score of', best_score, 'obtained with', best_hyperparams)

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    english_session_sentences = get_session_sentences(args.speaker_info, args.english_file)
    german_session_sentences = get_session_sentences(args.speaker_info, args.german_file)
    english_original, translated_to_english, german_original, translated_to_german = get_original_and_translated_sentences(english_session_sentences, german_session_sentences)
    # Using English data
    #full_english_session_data = get_concatenated_session_contents(english_session_sentences)
    #X_english_train, X_english_test = train_test_split(full_english_session_data, test_size=0.2, random_state=0)
    #optimise_num_of_topics(X_english_train, get_topic_feature_age_vectors, 'all_english', 'en', splits=3)
    #optimise_num_of_topics(X_english_train, get_topic_feature_gender_vectors, 'all_english', 'en', splits=3)
    # Using original language data
    # English
    original_english_session_data = get_concatenated_session_contents(english_original)
    print('Found', len(original_english_session_data), 'session contents with English as original language')
    X_orig_english_train, X_orig_english_test = train_test_split(original_english_session_data, test_size=0.2, random_state=0)
    optimise_num_of_topics(X_orig_english_train, get_topic_feature_gender_vectors, 'orig_english', 'en', splits=3)
    # German
    original_german_session_data = get_concatenated_session_contents(german_original)
    print('Found', len(original_german_session_data), 'session contents with German as original language')
    X_orig_german_train, X_orig_german_test = train_test_split(original_german_session_data, test_size=0.2, random_state=0)
    optimise_num_of_topics(X_orig_german_train, get_topic_feature_gender_vectors, 'orig_german', 'de', splits=3)
    # Using translated language data
    # English
    translated_to_english_session_data = get_concatenated_session_contents(translated_to_english)
    print('Found', len(translated_to_english_session_data), 'session contents with English as original language')
    X_trans_english_train, X_trans_english_test = train_test_split(translated_to_english_session_data, test_size=0.2, random_state=0)
    optimise_num_of_topics(X_trans_english_train, get_topic_feature_gender_vectors, 'trans_english', 'en', splits=3)
    # German
    translated_to_german_session_data = get_concatenated_session_contents(translated_to_german)
    print('Found', len(translated_to_german_session_data), 'session contents with German as original language')
    X_trans_german_train, X_trans_german_test = train_test_split(translated_to_german_session_data, test_size=0.2, random_state=0)
    optimise_num_of_topics(X_trans_german_train, get_topic_feature_gender_vectors, 'trans_german', 'de', splits=3)

