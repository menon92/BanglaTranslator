from . import utils
from . import models

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

from matplotlib import ticker
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

FONT_NAME = 'assets/banglafonts/Siyamrupali.ttf'

class Infer():
    def __init__(self, input_language_tokenizer, target_language_tokenizer,
                max_length_input, max_length_target, encoder, decoder, units):
        self.input_language_tokenizer = input_language_tokenizer
        self.target_language_tokenizer = target_language_tokenizer
        self.max_length_input = max_length_input
        self.max_length_target = max_length_target
        self.encoder = encoder
        self.decoder = decoder
        self.units = units
    
    def preprocess(self, sentence):
        # clean and pad sequece
        sentence = utils.clean_seq(sentence)
        sentence = utils.add_start_and_end_token_to_seq(sentence)
        
        inputs = [
            self.input_language_tokenizer.word_index[i] for i in sentence.split(' ')]
        inputs = sequence.pad_sequences(
            [inputs], maxlen=self.max_length_input,padding='post')
        tensor = tf.convert_to_tensor(inputs)

        return tensor

    def predict(self, sentence):
        tensor = self.preprocess(sentence)

        # init encoder
        encoder_initial_hidden = [tf.zeros((1, self.units))]
        encoder_out, encoder_hidden = self.encoder(tensor, encoder_initial_hidden)

        # init decoder
        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims(
            [self.target_language_tokenizer.word_index['<start>']], 0)

        result = ''
        for _ in range(self.max_length_target):
            predictions, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_out)
            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.target_language_tokenizer.index_word[predicted_id] + ' '
            if self.target_language_tokenizer.index_word[predicted_id] == '<end>':
                return result
            # the predicted ID is fed back into the model insteqad of using 
            # teacher forcing that we use in training time
            decoder_input = tf.expand_dims([predicted_id], 0)

        return result

    def predict_with_attention_weights(self, sentence):
        tensor = self.preprocess(sentence)

        # init encoder
        encoder_initial_hidden = [tf.zeros((1, self.units))]
        encoder_out, encoder_hidden = self.encoder(tensor, encoder_initial_hidden)

        # init decoder
        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims(
            [self.target_language_tokenizer.word_index['<start>']], 0)

        result = ''
        attention_plot = np.zeros((self.max_length_target, self.max_length_input))
        for t in range(self.max_length_target):
            predictions, decoder_hidden, attention_weights = \
                self.decoder(decoder_input, decoder_hidden, encoder_out)
            
            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.target_language_tokenizer.index_word[predicted_id] + ' '
            if self.target_language_tokenizer.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model insteqad of using 
            # teacher forcing that we use in training time
            decoder_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    prop = fm.FontProperties(fname=FONT_NAME)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    ax.set_xticklabels([''] + sentence, rotation=90, fontproperties=prop)
    ax.set_yticklabels([''] + predicted_sentence, fontproperties=prop)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.rcParams.update({'font.size': 14})

    plt.show()