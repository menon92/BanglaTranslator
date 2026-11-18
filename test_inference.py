"""Smoke test: restore the latest checkpoint and translate sentences from the training data."""
import tensorflow as tf

from translator import config as cfg
from translator import infer, models, utils

inp_tok = utils.load_tokenizer('models/input_language_tokenizer.json')
tgt_tok = utils.load_tokenizer('models/target_language_tokenizer.json')
encoder = models.Encoder(len(inp_tok.word_index) + 1, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)
decoder = models.Decoder(len(tgt_tok.word_index) + 1, cfg.EMBEDDING_DIM, cfg.UNITS, cfg.BATCH_SIZE)

# Keras 3 cannot restore into unbuilt layers, so run one dummy forward pass first
_out, _h = encoder(tf.zeros((1, cfg.MAX_INPUT_LANG_LEN), tf.int32), [tf.zeros((1, cfg.UNITS))])
decoder(tf.zeros((1, 1), tf.int32), _h, _out)

latest = tf.train.latest_checkpoint('models/training_checkpoints')
assert latest, 'no checkpoint found in models/training_checkpoints'
print('Restoring', latest)
tf.train.Checkpoint(encoder=encoder, decoder=decoder).restore(latest).expect_partial()

predictor = infer.Infer(inp_tok, tgt_tok, cfg.MAX_INPUT_LANG_LEN, cfg.MAX_TARGET_LANG_LEN,
                        encoder, decoder, cfg.UNITS)

# sentences taken from the first 3000 training pairs (bn -> expected en)
with open('data/ben-eng/ben.txt', encoding='utf8') as _f:
    all_lines = _f.read().strip().split('\n')
lines = all_lines[:3000]
samples = [l.split('\t')[:2][::-1] for l in lines[::300]]
ok = 0
for bn, en in samples:
    out = predictor.predict(bn).replace('<end>', '').strip()
    exp = utils.clean_seq(en)
    ok += out == exp
    print(f'{bn}\n  expected: {exp}\n  got:      {out}')
print(f'exact matches: {ok}/{len(samples)}')

try:
    predictor.predict('zzzunknown')
except ValueError as e:
    print('OOV handled:', e)

# held-out check: pairs after the first 3000 (never trained on) whose Bangla words are all known
held = [l.split('\t')[:2][::-1] for l in all_lines[3000:]]
known = [(bn, en) for bn, en in held
         if all(w in inp_tok.word_index for w in utils.clean_seq(bn).split(' '))]
print(f'\nheld-out pairs fully in vocabulary: {len(known)}/{len(held)}')
hit = 0
for bn, en in known[:200]:
    out = predictor.predict(bn).replace('<end>', '').strip()
    hit += out == utils.clean_seq(en)
for bn, en in known[:8]:
    print(f'{bn}\n  expected: {utils.clean_seq(en)}\n  got:      {predictor.predict(bn).replace("<end>", "").strip()}')
print(f'held-out exact matches: {hit}/{min(200, len(known))}')
