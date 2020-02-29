import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
    parser.add_argument("-i", "--speaker-info", type=str, default="europarl/europarl.fr-en/europarl.fr-en.dat",
                        help="Path to the file with speaker info")
    return parser

