import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description='Obtain SIFT features for training set')
    parser.add_argument("-i", "--speaker-info", type=str, default="europarl/europarl.de-en/europarl.de-en.dat",
                        help="Path to the file with speaker info")
    parser.add_argument("-g", "--german-file", type=str, default="europarl/europarl.de-en/europarl.de-en.de.aligned.tok",
                        help="Path to the file with the aligned german phrases")
    parser.add_argument("-e", "--english-file", type=str, default="europarl/europarl.de-en/europarl.de-en.en.aligned.tok",
                        help="Path to the file with the aligned english phrases")
    return parser

