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

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    raw_speaker_info = utils.read_file_lines(args.speaker_info)
    speaker_info = build_speaker_info(raw_speaker_info)
    import ipdb; ipdb.set_trace()