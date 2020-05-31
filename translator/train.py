import os
import time
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from . import config as cfg
from . import datasets
from . import utils
from . import models


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, targ_lang_tokenizer, 
            enc_hidden, encoder, decoder, optimizer):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [targ_lang_tokenizer.word_index['<start>']]*cfg.BATCH_SIZE, 1)
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def run():
    text_data = datasets.TatoebaDataset('data/ben-eng/ben.txt', cfg.NUM_DATA_TO_LOAD)
    
    # retrive data and tokenizers
    tensors, tokenizer = text_data.load_data()
    input_tensor, target_tensor = tensors 
    inp_lang_tokenizer, targ_lang_tokenizer = tokenizer

    # save tokenizer for further use
    utils.save_tokenizer(
        tokenizer=inp_lang_tokenizer,
        save_at='models',
        file_name='input_language_tokenizer.json')
    utils.save_tokenizer(
        tokenizer=targ_lang_tokenizer,
        save_at='models',
        file_name='target_language_tokenizer.json')  

    # Creating training and validation sets using an 80-20 split
    input_train, input_val, target_train, target_val = \
        train_test_split(input_tensor, target_tensor, test_size=0.2)

    # set training params
    buffer_size = len(input_train)
    steps_per_epoch = len(input_train) // cfg.BATCH_SIZE
    vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1
    vocab_tar_size = len(targ_lang_tokenizer.word_index) + 1

    # convert data to tf.data formate
    dataset = tf.data.Dataset.from_tensor_slices((input_train, target_train))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)

    # init optimizer
    optimizer = tf.keras.optimizers.Adam()
    
    # init encoder & decoder
    encoder = models.Encoder(
        vocab_inp_size, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
    decoder = models.Decoder(
        vocab_tar_size, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)

    # init checkpoint 
    checkpoint_dir = 'models/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)

    if cfg.RESTORE_SAVED_CHECKPOINT:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for epoch in range(cfg.EPOCHS):
        print("Epoch {} / {}".format(epoch, cfg.EPOCHS))
        pbar = tqdm(dataset.take(steps_per_epoch), ascii=True)
        
        total_loss = 0
        enc_hidden = encoder.initialize_hidden_state()
        
        for step, data in enumerate(pbar):
            inp, targ = data
            batch_loss = train_step(
                inp, targ, targ_lang_tokenizer,
                enc_hidden, encoder, decoder, optimizer)
            
            total_loss += batch_loss

            pbar.set_description(
                "Step - {} / {} - batch loss - {:.4f} - "
                    .format(steps_per_epoch, step+1, batch_loss.numpy()))
        
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch loss - {:.4f}'.format(total_loss / steps_per_epoch))

if __name__ == "__main__":
    run()
