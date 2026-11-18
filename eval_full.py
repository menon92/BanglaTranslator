"""Evaluate models_v2 on the held-out validation sentences (same split as train_full.py)."""
import random

import tensorflow as tf

from translator import config as cfg
from translator import infer, models, utils

MAXLEN, OUT = 16, 'models_v2'
random.seed(0)
pairs = []
with open('data/ben-eng-v2/ben.txt', encoding='utf8') as _f:
    _lines = _f.read().strip().split('\n')
for line in _lines:
    en, bn = line.split('\t')[:2]
    en, bn = utils.add_start_and_end_token_to_seq(utils.clean_seq(en, True)), utils.add_start_and_end_token_to_seq(utils.clean_seq(bn, True))
    if len(en.split()) <= MAXLEN and len(bn.split()) <= MAXLEN:
        pairs.append((bn, en))
srcs = sorted({b for b, _ in pairs}); random.shuffle(srcs)
val_src = set(srcs[:int(0.1 * len(srcs))]); val = [p for p in pairs if p[0] in val_src]

inp_tok = utils.load_tokenizer(f'{OUT}/input_language_tokenizer.json'); tgt_tok = utils.load_tokenizer(f'{OUT}/target_language_tokenizer.json')
enc = models.Encoder(len(inp_tok.word_index) + 1, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
dec = models.Decoder(len(tgt_tok.word_index) + 1, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
o, h = enc(tf.zeros((1, MAXLEN), tf.int32), [tf.zeros((1, cfg.UNITS))]); dec(tf.zeros((1, 1), tf.int32), h, o)
tf.train.Checkpoint(encoder=enc, decoder=dec).read(f'{OUT}/best').expect_partial()
pred = infer.Infer(inp_tok, tgt_tok, MAXLEN, MAXLEN, enc, dec, cfg.UNITS, keep_marks=True)

def strip(s): return s.replace('<start>', '').replace('<end>', '').strip()
hit = tot = unk = 0; shown = 0
for bn, en in val[:150]:
    words = strip(bn).split()
    kept = [w for w in words if w in inp_tok.word_index]   # drop unseen words, as in training
    unk += len(kept) < len(words)
    out = strip(pred.predict(' '.join(kept)))
    tot += 1; hit += out == strip(en)
    if shown < 12:
        print(strip(bn)); print('  expected:', strip(en)); print('  got:     ', out); shown += 1
print(f'val pairs {tot} (with unseen words: {unk}); exact match {hit}/{tot} = {hit/tot:.1%}')
