import os
import numpy as np

from slacker import Slacker


SLACK_TOKEN = 'xoxp-554173958562-554173959170-555244937223-1f3cfc06ff8cc48d3a2ea00e6c682a7c'

def initiate_slacker():
    """
    Initiate slack object
    return :
        slack object
    """
    slack = Slacker(SLACK_TOKEN)

    if slack.api.test().successful:
        print(f"Connected to {slack.team.info().body['team']['name']}.")
    else:
        print('Try Again!')

    return slack

def slack_post_message(slack, text, channel, username):
    """
    Post message in slack channel
    """
    r = slack.chat.post_message(channel=channel, text=text,
                                username=username,
                                icon_emoji=':running:')

def open_pickle(path):
    """
    Load pickle file
    path : relative path to the pickle.file
    return : unpickled file
    """
    import pickle
    with open(path, 'rb') as f:
        X = pickle.load(f)
    return X

def extract_glove_index(GLOVE_DIR, file):
    """
    Extract pre-trained GLOVE matrices
    Param :
        GLOVE_DIR path to the directory
        file    relative path to the file
    """
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, file), 'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index