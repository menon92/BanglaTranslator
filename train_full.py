"""Retrain on the larger manythings.org dataset with a held-out validation set.
Writes to models_v2/ so the original model in models/ is untouched."""
import os
import random
import sys
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

from translator import config as cfg
from translator import models, utils
from translator.train import loss_function, train_step

MAXLEN, EPOCHS, OUT = 16, int(sys.argv[1]) if len(sys.argv) > 1 else 30, 'models_v2'
random.seed(0); tf.random.set_seed(0)
os.makedirs(OUT, exist_ok=True)

pairs = []
with open('data/ben-eng-v2/ben.txt', encoding='utf8') as _f:
    _lines = _f.read().strip().split('\n')
for line in _lines:
    en, bn = line.split('\t')[:2]
    en, bn = utils.add_start_and_end_token_to_seq(utils.clean_seq(en, True)), utils.add_start_and_end_token_to_seq(utils.clean_seq(bn, True))
    if len(en.split()) <= MAXLEN and len(bn.split()) <= MAXLEN:
        pairs.append((bn, en))

# split by unique Bangla source so no source sentence appears in both sets
srcs = sorted({b for b, _ in pairs}); random.shuffle(srcs)
val_src = set(srcs[:int(0.1 * len(srcs))])
train = [p for p in pairs if p[0] not in val_src]; val = [p for p in pairs if p[0] in val_src]
print(f'train {len(train)}  val {len(val)}')

inp_tok = utils.get_lang_tokenize([b for b, _ in train]); tgt_tok = utils.get_lang_tokenize([e for _, e in train])
utils.save_tokenizer(inp_tok, OUT, 'input_language_tokenizer.json'); utils.save_tokenizer(tgt_tok, OUT, 'target_language_tokenizer.json')
def tens(pp):
    a = sequence.pad_sequences(inp_tok.texts_to_sequences([b for b, _ in pp]), maxlen=MAXLEN, padding='post')
    b = sequence.pad_sequences(tgt_tok.texts_to_sequences([e for _, e in pp]), maxlen=MAXLEN, padding='post')
    return a, b
(xt, yt), (xv, yv) = tens(train), tens(val)

enc = models.Encoder(len(inp_tok.word_index) + 1, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
dec = models.Decoder(len(tgt_tok.word_index) + 1, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
opt = tf.keras.optimizers.Adam()
ds = tf.data.Dataset.from_tensor_slices((xt, yt)).shuffle(len(xt)).batch(cfg.BATCH_SIZE, drop_remainder=True)
steps = len(xt) // cfg.BATCH_SIZE
start = tgt_tok.word_index['<start>']

def val_loss():
    tot = n = 0
    for i in range(0, len(xv) - cfg.BATCH_SIZE + 1, cfg.BATCH_SIZE):
        inp, targ = tf.constant(xv[i:i+cfg.BATCH_SIZE]), tf.constant(yv[i:i+cfg.BATCH_SIZE])
        o, h = enc(inp, enc.initialize_hidden_state()); d = tf.expand_dims([start] * cfg.BATCH_SIZE, 1); l = 0
        for t in range(1, targ.shape[1]):
            p, h, _ = dec(d, h, o); l += loss_function(targ[:, t], p); d = tf.expand_dims(targ[:, t], 1)
        tot += float(l) / targ.shape[1]; n += 1
    return tot / n

ckpt = tf.train.Checkpoint(encoder=enc, decoder=dec)
best, bad = 1e9, 0
for ep in range(EPOCHS):
    t0 = time.time(); tl = 0
    for inp, targ in ds.take(steps):
        tl += float(train_step(inp, targ, tgt_tok, enc.initialize_hidden_state(), enc, dec, opt))
    vl = val_loss()
    print(f'epoch {ep+1}/{EPOCHS} train {tl/steps:.4f} val {vl:.4f} ({time.time()-t0:.0f}s)', flush=True)
    if vl < best:
        best, bad = vl, 0; ckpt.write(os.path.join(OUT, 'best')); print('  saved best', flush=True)
    else:
        bad += 1
        if bad >= 4: print('early stop'); break
