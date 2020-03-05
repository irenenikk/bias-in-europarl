from data_pipeline import get_session_sentences, \
                            get_concatenated_session_contents, \
                            preprocess_docs,\
                            get_original_and_translated_sentences
from argparser import get_argparser
from topic_modelling import get_topic_feature_gender_vectors, get_topic_feature_country_vectors, get_topic_feature_age_vectors
from classifier import evaluate_svm

def obtain_svm_results(data, num_topics, model_id, lang, **kwargs):
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)
    train_features, train_labels = get_topic_feature_gender_vectors(X_train, num_topics, model_id, lang)
    test_features, test_labels = get_topic_feature_gender_vectors(X_test, num_topics, model_id, lang)
    acc, f1_score = evaluate_svm(train_features, train_labels, test_features, test_labels, **kwargs)
    return acc, f1_score

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    english_session_sentences = get_session_sentences(args.speaker_info, args.english_file)
    german_session_sentences = get_session_sentences(args.speaker_info, args.german_file)
    english_original, translated_to_english, german_original, translated_to_german = get_original_and_translated_sentences(english_session_sentences, german_session_sentences)
    # Using English data
    full_english_session_data = get_concatenated_session_contents(english_session_sentences)
    print('Found', len(full_english_session_data), 'session contents in total in English')
    # TODO: run hyperparameter training to choose best svm parameters
    acc, f1_score = obtain_svm_results(full_english_session_data, 100, 'all_english', 'en', kernel='rbf', class_weight='balanced')
    print('Acc', acc, 'F1 score', f1_score, 'in gender prediction based on topics')
    # Using original language data
    # English
    original_english_session_data = get_concatenated_session_contents(english_original)
    print('Found', len(original_english_session_data), 'session contents with English as original language')
    acc, f1_score = obtain_svm_results(original_english_session_data, 100, 'orig_english', 'en', kernel='rbf', class_weight='balanced')
    print('Acc', acc, 'F1 score', f1_score, 'in gender prediction based on topics')
    # German
    original_german_session_data = get_concatenated_session_contents(german_original)
    print('Found', len(original_german_session_data), 'session contents with German as original language')
    acc, f1_score = obtain_svm_results(original_german_session_data, 100, 'orig_german', 'de_core_news_sm', kernel='rbf', class_weight='balanced')
    print('Acc', acc, 'F1 score', f1_score, 'in gender prediction based on topics')
    # TODO: get features for training and test sets and evaluate model
    # Using translated language data
    # English
    translated_to_english_session_data = get_concatenated_session_contents(translated_to_english)
    print('Found', len(translated_to_english_session_data), 'session contents with English as original language')
    acc, f1_score = obtain_svm_results(translated_to_english_session_data, 100, 'trans_english', 'en', kernel='rbf', class_weight='balanced')
    print('Acc', acc, 'F1 score', f1_score, 'in gender prediction based on topics')
    # German
    translated_to_german_session_data = get_concatenated_session_contents(translated_to_german)
    print('Found', len(translated_to_german_session_data), 'session contents with German as original language')
    acc, f1_score = obtain_svm_results(translated_to_german_session_data, 100, 'trans_german', 'de_core_news_sm', kernel='rbf', class_weight='balanced')
    print('Acc', acc, 'F1 score', f1_score, 'in gender prediction based on topics')
    # TODO: get features for training and test sets and evaluate model

