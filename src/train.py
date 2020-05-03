import os
import time
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import config as cfg
import datasets
import utils
import models

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
    text_data = datasets.TatoebaDataset('../data/ben-eng/ben.txt', cfg.NUM_DATA_TO_LOAD)
    
    tensors, tokenizer = text_data.load_data()
    input_tensor, target_tensor = tensors 
    inp_lang_tokenizer, targ_lang_tokenizer = tokenizer

    print('-' * 45)
    print('input tensor shape : {}'.format(input_tensor.shape))
    print('target tensor shape: {}'.format(target_tensor.shape))
    print("input tensor : {}".format(input_tensor[-1]))
    print('terget tensor: {}'.format(target_tensor[-1]))

    utils.show_index_to_word_maping(inp_lang_tokenizer, input_tensor[-1])
    utils.show_index_to_word_maping(targ_lang_tokenizer, target_tensor[-1])    

    # Creating training and validation sets using an 80-20 split
    input_train, input_val, target_train, target_val = train_test_split(
        input_tensor, target_tensor, test_size=0.2)

    print('-' * 45)
    print("Input train data : {}".format(len(input_train)))
    print("Target train data: {}".format(len(target_train)))
    print('Input valid data : {}'.format(len(input_val)))
    print('Target valid data: {}'.format(len(target_val)))

    buffer_size = len(input_train)
    steps_per_epoch = len(input_train)//cfg.BATCH_SIZE
    vocab_inp_size = len(inp_lang_tokenizer.word_index)+1
    vocab_tar_size = len(targ_lang_tokenizer.word_index)+1

    print('-' * 45)
    print('buffer size      : {}'.format(buffer_size))
    print('steps per epochs : {}'.format(steps_per_epoch))
    print('vocab input size : {}'.format(vocab_inp_size))
    print('vocab target size: {}'.format(vocab_tar_size))

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_train, target_train)).shuffle(buffer_size)
    dataset = dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)


    example_input_batch, example_target_batch = next(iter(dataset))
    print('input shape :', example_input_batch.shape)
    print('target shaep:', example_target_batch.shape)
    print(example_input_batch[:2])
    print(example_input_batch[:2])

    optimizer = tf.keras.optimizers.Adam()
    
    encoder = models.Encoder(
        vocab_inp_size, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
    decoder = models.Decoder(
        vocab_tar_size, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)

    checkpoint_dir = '../models/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)

    for epoch in range(cfg.EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, targ_lang_tokenizer,
                 enc_hidden, encoder, decoder, optimizer)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

if __name__ == "__main__":
    run()
