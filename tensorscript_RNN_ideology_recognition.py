import h5py
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json

# The maximum number of words to be used. (most frequent)
max_top_words = 50000
# Max number of words in each complaint.
max_tweet_lenght = 300

tokenizer = Tokenizer(num_words=max_top_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

tweet = input("Enter your tweet here: ")
tweet = [str(tweet)]

seq = tokenizer.texts_to_sequences(tweet)
padded = pad_sequences(seq, maxlen=max_tweet_lenght)
pred = loaded_model.predict(padded)
labels=['populistic','technocratic']
print(pred, labels[np.argmax(pred)])
