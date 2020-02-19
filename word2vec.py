import os
import json
import pickle as pkl
from collections import defaultdict, Counter

import numpy as np
import pytrec_eval
from tqdm import tqdm

import read_ap
import download_ap

from SkipGram import SkipGram

if __name__ == "__main__":

    # ensure dataset is downloaded
    download_ap.download_dataset()
    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # Create instance for retrieval
    skip_gram = SkipGram()
