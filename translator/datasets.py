import io
from . import utils

class TatoebaDataset():
    def __init__(self, path_to_file, num_data_to_load):
        self.path_to_file = path_to_file
        self.num_data_to_load = num_data_to_load

    def read_data(self):
        lines = io.open(
            self.path_to_file, encoding='UTF-8').read().strip().split('\n')
        return lines

    def make_sequence_pair(self, lines):
        seq_pairs = []
        for line in lines[:self.num_data_to_load]:
            en, bn, _ = line.split('\t')
            pair = []
            for seq in [en, bn]:
                seq = utils.clean_seq(seq)
                seq = utils.add_start_and_end_token_to_seq(seq)
                pair.append(seq)    
            seq_pairs.append(pair)
        return seq_pairs

    def create_dataset(self):
        lines = self.read_data()
        word_pairs = self.make_sequence_pair(lines)
        return zip(*word_pairs)
    
    def load_data(self):
        # creating cleaned input, output pairs
        targ_lang_text, inp_lang_text = self.create_dataset()

        targ_lang_tokenizer = utils.get_lang_tokenize(targ_lang_text)
        inp_lang_tokenizer = utils.get_lang_tokenize(inp_lang_text)
        
        target_tensor = utils.texts_to_sequences(targ_lang_text, targ_lang_tokenizer)
        input_tensor  = utils.texts_to_sequences(inp_lang_text, inp_lang_tokenizer)
       
        tensor_pair = (input_tensor, target_tensor)
        tokenizer_pair = (inp_lang_tokenizer, targ_lang_tokenizer)

        return tensor_pair, tokenizer_pair