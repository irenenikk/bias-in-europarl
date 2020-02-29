import re
from argparser import get_argparser
import utils
from collections import defaultdict
import pandas as pd

def build_speaker_info(raw_speaker_info):
    attributes = ['COUNT', 'EUROID', 'NAME', 'LANGUAGE', 'GENDER', 'DATE_OF_BIRTH', 'SESSION_DATE', 'AGE']
    speaker_dict = defaultdict(list)
    for att in attributes:
        for info in raw_speaker_info:
            value = re.search(rf" {att}=\"(.*?)\"", info).group(1)
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
    for name, group in grouped_by_session_and_id:
        session_content = ' '.join(group['sentence'].values)
        session_dict['content'].append(session_content)
        cols_to_skip = ['sentence', 'count', 'language', 'name']
        for col in group.columns:
            if col in cols_to_skip:
                continue
            session_dict[col].append(group.iloc[0][col])
    session_contents = pd.DataFrame.from_dict(session_dict)
    return session_contents

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    mep_details = pd.read_csv(args.mep_details, sep='\t')
    session_sentences = get_session_sentences(args.speaker_info, args.target_lang)
    full_session_data = get_concatenated_session_contents(session_sentences)
    import ipdb; ipdb.set_trace()
    