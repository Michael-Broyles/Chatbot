# Super Data Science YouTube Tutorial
# Recurrent Neural Network - Sequence to Sequence Model: Encoder & Decoder - Long Short Term Memory
from __future__ import absolute_import, division, print_function

import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *


path = '../chatbot_data/einstein_input.txt'
char_idx_file = 'einstein_pickle/char_idx.pickle'

# if not os.path.isfile(path):
#     urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/shakespeare_input.txt")

# Max length ???
maxlen = 25

char_idx = None
if os.path.isfile(char_idx_file):
    print('Loading previous char_idx')
    char_idx = pickle.load(open(char_idx_file, 'rb'))

X, Y, char_idx = textfile_to_semi_redundant_sequences(path,
                                                      seq_maxlen=maxlen,
                                                      redun_step=3,
                                                      pre_defined_char_idx=char_idx)

pickle.dump(char_idx, open(char_idx_file, 'wb'))


# Creates the LSTM model
g = tflearn.input_data([None, maxlen, len(char_idx)])# Input Layer

g = tflearn.lstm(g, 128, return_seq=True)# Layer 1
g = tflearn.dropout(g, 0.5)

g = tflearn.lstm(g, 128, return_seq=True)# Layer 2
g = tflearn.dropout(g, 0.5)

g = tflearn.lstm(g, 128, return_seq=True)# Layer 3
g = tflearn.dropout(g, 0.5)

g = tflearn.lstm(g, 128, return_seq=True)# Layer 4
g = tflearn.dropout(g, 0.5)

g = tflearn.lstm(g, 128, return_seq=True)# Layer 5
g = tflearn.dropout(g, 0.5)

g = tflearn.lstm(g, 128)# Layer 6
g = tflearn.dropout(g, 0.5)

g = tflearn.fully_connected(g, len(char_idx), activation='softmax')

g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       metric='accuracy', learning_rate=0.001)

# What is 'tflearn.SequenceGenerator()' ???
model = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='trained_models/einstein/einstein_model',
                              tensorboard_dir='tensorboard_logs')


# Runs forwards & Backwards
for i in range(2):
    seed = random_sequence_from_textfile(path, maxlen)
    model.fit(X, Y, validation_set=0.1, batch_size=128, n_epoch=10, run_id = 'einstein')

    print('-- TESTING...')
    print('-- Test with temperature of 1.0 --')
    print(model.generate(600, temperature=1.0, seq_seed=seed))# temperature=1.0
    print('-- Test with temperature of 0.5 --')
    print(model.generate(600, temperature=0.5, seq_seed=seed))# temperature=0.5





# tensorboard --path=tensorboard_logs/shakespeare

