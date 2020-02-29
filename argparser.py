import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
    parser.add_argument("-i", "--speaker-info", type=str, default="europarl/europarl.fr-en/europarl.fr-en.dat",
                        help="Path to the file with speaker info")
    parser.add_argument("-s", "--source-lang", type=str, default="europarl/europarl.fr-en/europarl.fr-en.fr.aligned.tok",
                        help="Path to the file with the aligned source phrases")
    parser.add_argument("-t", "--target-lang", type=str, default="europarl/europarl.fr-en/europarl.fr-en.en.aligned.tok",
                        help="Path to the file with the aligned target phrases")
    parser.add_argument("-meps", "--mep-details", type=str, default="europarl/MEPs.details.txt",
                        help="Path to the file with details for each MEP")
    return parser

