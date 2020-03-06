import heapq
import itertools
import os
import pickle
import sys
from collections import defaultdict

import kenlm
import nltk
import numpy as np

import datrie
from paths import paths
from tokenization import tokenize_mid_document
from util import logsumexp

LOG10 = np.log(10)


def get_arpa_bigrams(filename):
    with open(filename) as f:
        while not f.readline().startswith('\\2-grams:'):
            continue
        bigrams = defaultdict(list)
        for line in f:
            line = line.strip()
            if not line:
                break  # end of 2-grams
            parts = line.split('\t')
            prob = float(parts[0])
            a, b = parts[1].split(' ')
            bigrams[a].append((prob, b))
        return bigrams


def encode_bigrams(bigrams, model):
    id2str = {}
    encoded_bigrams = {}
    for prev, nexts in bigrams.items():
        prev_id = model.vocab_index(prev)
        id2str[prev_id] = prev
        next_ids = []
        for prob, b in nexts:
            next_id = model.vocab_index(b)
            id2str[next_id] = b
            next_ids.append((prob, next_id))
        encoded_bigrams[prev_id] = next_ids
    def pull_2nd(lst):
        return [x[1] for x in lst]
    unfiltered_bigrams = {a: pull_2nd(nexts) for a, nexts in encoded_bigrams.items()}
    filtered_bigrams = {a: pull_2nd(heapq.nlargest(100, nexts)) for a, nexts in encoded_bigrams.items()}
    return id2str, unfiltered_bigrams, filtered_bigrams



class LanguageModel:
    def __init__(self, model_file, bigrams_file, arpa_file):
        print("Loading model", file=sys.stderr)
        self.model = kenlm.LanguageModel(model_file)
        print("...done.", file=sys.stderr)

        # Bigrams
        if os.path.exists(bigrams_file):
            print("Loading bigrams pickle", file=sys.stderr)
            self._bigrams = pickle.load(open(bigrams_file, 'rb'))
        else:
            print("Reading ARPA bigrams", file=sys.stderr)
            bigrams = get_arpa_bigrams(arpa_file)
            print("Encoding bigrams to indices", file=sys.stderr)
            self._bigrams = encode_bigrams(bigrams, self.model)
            print("Saving bigrams", file=sys.stderr)
            with open(bigrams_file, 'wb') as f:
                pickle.dump(self._bigrams, f, -1)

        id2str_dict, self.unfiltered_bigrams, self.filtered_bigrams = self._bigrams
        self.id2str = [None] * (max(id2str_dict.keys()) + 1)
        for i, s in id2str_dict.items():
            self.id2str[i] = s

        # Vocab trie
        self.vocab_trie = datrie.BaseTrie(set(itertools.chain.from_iterable(id2str_dict.values())))
        for i, s in id2str_dict.items():
            self.vocab_trie[s] = i

        self.eos = '</S>'
        self.eop = '</s>'

    def prune_bigrams(self):
        # Filter bigrams to only include words that actually follow
        bigrams = self.unfiltered_bigrams
        while True:
            new_bigrams = {k: [tok for tok in v if len(bigrams.get(tok, [])) > 0] for k, v in bigrams.items()}
            new_bigrams_trim = {k: v for k, v in new_bigrams.items() if len(v) > 0}
            if len(new_bigrams) == len(new_bigrams_trim):
                break
            bigrams = new_bigrams_trim
        self.unfiltered_bigrams = bigrams

    def _compute_pos(self):
        print("Computing pos tags")
        pos_tags = [nltk.pos_tag([w or "UNK"], tagset='universal')[0][1] for w in self.id2str]
        self._id2tag = sorted(set(pos_tags))
        tag2id = {tag: id for id, tag in enumerate(self._id2tag)}
        self._pos_tags = np.array([tag2id[tag] for tag in pos_tags])

    @property
    def pos_tags(self):
        if not hasattr(self, '_pos_tags'):
            self._compute_pos()
        return self._pos_tags

    @property
    def id2tag(self):
        if not hasattr(self, '_id2tag'):
            self._compute_pos()
        return self._id2tag

    @property
    def word_lengths(self):
        if not hasattr(self, '_word_lengths'):
            self._word_lengths = np.array([len(w) if w is not None else 0 for w in self.id2str])
        return self._word_lengths


    @classmethod
    def from_basename(cls, basename):
        return cls(model_file=basename + '.kenlm', bigrams_file=basename + '.bigrams.pkl', arpa_file=basename + '.arpa')

    @property
    def bos_state(self):
        state = kenlm.State()
        self.model.BeginSentenceWrite(state)
        return state

    @property
    def null_context_state(self):
        state = kenlm.State()
        self.model.NullContextWrite(state)
        return state

    def get_state(self, words, bos=False):
        if bos:
            state = self.bos_state
        else:
            state = self.null_context_state
        score, state = self.score_seq(state, words)
        return state, score

    def score_seq(self, state, words):
        score = 0.
        for word in words:
            new_state = kenlm.State()
            score += self.model.BaseScore(state, word, new_state)
            state = new_state
        return score * LOG10, state

    def score_seq_by_word(self, state, words):
        scores = []
        for word in words:
            new_state = kenlm.State()
            scores.append(LOG10 * self.model.BaseScore(state, word, new_state))
            state = new_state
        return scores

    def next_word_logprobs_raw(self, state, prev_word, prefix_logprobs=None):
        bigrams = self.unfiltered_bigrams
        if prefix_logprobs is not None:
            next_words = []
            prior_logprobs = []
            for logprob, prefix in prefix_logprobs:
                for word, word_idx in self.vocab_trie.items(prefix):
                    next_words.append(word_idx)
                    prior_logprobs.append(logprob)
        else:
            next_words = bigrams.get(self.model.vocab_index(prev_word), [])
            if len(next_words) == 0:
                next_words = bigrams.get(self.model.vocab_index('<S>'), [])
            next_words = [w for w in next_words if w != self.eos and w != self.eop]
        if len(next_words) == 0:
            return [], np.zeros(0)
        new_state = kenlm.State()
        logprobs = np.empty(len(next_words))
        for next_idx, word_idx in enumerate(next_words):
            logprob = self.model.base_score_from_idx(state, word_idx, new_state)
            if prefix_logprobs is not None:
                logprob += prior_logprobs[next_idx]
            logprobs[next_idx] = logprob
        logprobs *= LOG10
        return next_words, logprobs


# Temp alias
Model = LanguageModel

models = {name: Model.from_basename(paths.model_basename(name)) for name in ['yelp_train']}
def get_model(name):
    return models[name]



def softmax(scores):
    return np.exp(scores - logsumexp(scores))


#def sample_phrases(context_toks, n, temperature=.5):
#    '''draw n different phrases that all start with different words'''
#    probs = softmax(score_phrases_by_ngrams(context_toks) / temperature)
#    n_phrases = len(probs)
#    while True:
#        chosen_phrases = [idx2phrase[idx] for idx in np.random.choice(n_phrases, n, replace=False, p=probs)]
#        if True:#len({phrase[0] for phrase in chosen_phrases}) == n:
#            return chosen_phrases
#        print("Redraw")
#

def next_word_probs(model, state, prev_word, prefix_logprobs=None, temperature=1., length_bonus_min_length=6, length_bonus_amt=0., pos_weights=None):
    next_words, logprobs = model.next_word_logprobs_raw(state, prev_word, prefix_logprobs=prefix_logprobs)
    if len(next_words) == 0:
        return next_words, logprobs
    if length_bonus_amt:
        length_bonus_elegible = model.word_lengths[next_words] >= length_bonus_min_length
        logprobs = logprobs + length_bonus_amt * length_bonus_elegible
    if pos_weights is not None:
        poses = model.pos_tags[next_words]
        logprobs = logprobs + pos_weights[poses]
    logprobs /= temperature
    return next_words, softmax(logprobs)


class GenerationFailedException(Exception):
    pass

def retry_on_exception(exception, tries):
    def decorator(fn):
        def wrapper(*a, **kw):
            for i in range(tries):
                try:
                    return fn(*a, **kw)
                except exception:
                    continue
                except:
                    raise
            return fn(*a, **kw)
        return wrapper
    return decorator

@retry_on_exception(GenerationFailedException, 10)
def generate_phrase(model, context_toks, length, prefix_logprobs=None, **kw):
    if context_toks[0] == '<s>':
        state, _ = model.get_state(context_toks[1:], bos=True)
    else:
        state, _ = model.get_state(context_toks, bos=False)
    phrase = context_toks[:]
    generated_logprobs = np.empty(length)
    for i in range(length):
        next_words, probs = next_word_probs(model, state, phrase[-1], prefix_logprobs=prefix_logprobs, **kw)
        if len(next_words) == 0:
            raise GenerationFailedException
        prefix_logprobs = None
        picked_subidx = np.random.choice(len(probs), p=probs)
        picked_idx = next_words[picked_subidx]
        new_state = kenlm.State()
        model.model.base_score_from_idx(state, picked_idx, new_state)
        state = new_state
        word = model.id2str[picked_idx]
        phrase.append(word)
        generated_logprobs[i] = np.log(probs[picked_subidx])
    return phrase[len(context_toks):], generated_logprobs


def generate_diverse_phrases(model, context_toks, n, length, prefix_logprobs=None, **kw):
    if model is None:
        model = 'yelp_train'
    if isinstance(model, str):
        model = get_model(model)
    if 'pos_weights' in kw:
        kw['pos_weights'] = np.array(kw['pos_weights'])

    state, _ = model.get_state(context_toks)
    first_words, first_word_probs = next_word_probs(model, state, context_toks[-1], prefix_logprobs=prefix_logprobs, **kw)
    if len(first_words) == 0:
        return []
    res = []
    for idx in np.random.choice(len(first_words), min(len(first_words), n), p=first_word_probs, replace=False):
        first_word = model.id2str[first_words[idx]]
        first_word_logprob = np.log(first_word_probs[idx])
        phrase, phrase_logprobs = generate_phrase(model, context_toks + [first_word], length - 1, **kw)
        res.append(([first_word] + phrase, np.hstack(([first_word_logprob], phrase_logprobs))))
    return res


def beam_search_phrases(model, start_words, beam_width, length, prefix_probs=None, length_bonus=0):
    start_state, start_score = model.get_state(start_words)
    beam = [(start_score, [], False, start_state, None, 0)]
    for i in range(length):
        bigrams = model.unfiltered_bigrams if i == 0 else model.filtered_bigrams
        prefix_chars = 1 if i > 0 else 0
        def candidates():
            for score, words, done, penultimate_state, last_word, num_chars in beam:
                if done:
                    yield score, words, done, penultimate_state, last_word, num_chars
                    continue
                if last_word is not None:
                    last_state = kenlm.State()
                    model.model.base_score(penultimate_state, last_word, last_state)
                else:
                    last_state = penultimate_state
                probs = None
                if len(words) == 0 and prefix_probs is not None:
                    next_words = []
                    probs = []
                    for prob, prefix in prefix_probs:
                        for word, word in model.vocab_trie.items(prefix):
                            next_words.append(word)
                            probs.append(prob)
                else:
                    last_word = words[-1] if words else model.model.vocab_index(start_words[-1])
                    # print(id2str[last_word])
                    next_words = bigrams.get(last_word, [])
                new_state = kenlm.State()
                for next_idx, word in enumerate(next_words):
                    if word == '</S>' or word == "</s>":
                        continue
                    if probs is not None:
                        prob = probs[next_idx]
                    else:
                        prob = 0.
                    new_words = words + [word]
                    new_num_chars = num_chars + prefix_chars + len(word)
                    yield score + prob + word_score(word, length_bonus) + LOG10 * model.model.base_score(last_state, word, new_state), new_words, new_num_chars >= length, last_state, word, new_num_chars
        beam = heapq.nlargest(beam_width, candidates())
    return [dict(score=score, words=words, done=done, num_chars=num_chars) for score, words, done, _, _, num_chars in sorted(beam, reverse=True)]


BORING_WORDS = set('probably unfortunately definitely basically absolutely'.split())
def word_score(word, length_bonus):
    if word in BORING_WORDS:
        return .1
    return len(word) ** length_bonus
    # return 1 + len(word) * length_bonus

def tap_decoder(char_model, before_cursor, cur_word, key_rects, beam_width=100, scale=100.):
    keys = [k['key'] for k in key_rects]
    rects = [k['rect'] for k in key_rects]
    centers = [((rect['left'] + rect['right']) / 2, (rect['top'] + rect['bottom']) / 2) for rect in rects]

    beam_width = 100
    beam = [(0., '', None)]
    for item in cur_word:
        if 'tap' not in item:
            letter = item['letter']
            letters_and_distances = [(letter, 0)]
        else:
            x, y = item['tap']
            sq_dist_to_center = [(x - rect_x) ** 2. + (y - rect_y) ** 2. for rect_x, rect_y in centers]
            letters_and_distances = zip(keys, sq_dist_to_center)
        new_beam = []
        # print(np.min(sq_dist_to_center) / scale, keys[np.argmin(sq_dist_to_center)])
        for score, sofar, penultimate_state in beam:
            last_state = kenlm.State()
            if sofar:
                char_model.BaseScore(penultimate_state, sofar[-1], last_state)
            else:
                char_model.NullContextWrite(last_state)
                for c in before_cursor:
                    next_state = kenlm.State()
                    char_model.BaseScore(last_state, c, next_state)
                    last_state = next_state
            next_state = kenlm.State()
            for key, dist in letters_and_distances:
                new_so_far = sofar + key
                new_beam.append((score + char_model.BaseScore(last_state, key, next_state) - dist / scale, new_so_far, last_state))
        beam = sorted(new_beam, reverse=True)[:beam_width]
    return [(prob, word) for prob, word, state in sorted(beam, reverse=True)[:10]]


def tokenize_sofar(sofar):
    toks = tokenize_mid_document(sofar.lower().replace(' .', '.').replace(' ,', ','))[0]
    if toks[-1] != '':
        print("WEIRD: somehow we got a mid-word sofar:", repr(sofar))
    assert toks[0] == "<D>"
    assert toks[1] == "<P>"
    assert toks[2] == "<S>"
    return ['<s>', "<D>"] + toks[3:-1]


def get_touch_suggestions(sofar, cur_word, key_rects, beam_width=10, length_bonus=0):
    if len(cur_word) > 0:
        prefix_probs = [(1., ''.join(item['letter'] for item in cur_word))]
        # prefix_probs = tap_decoder(sofar[-12:].replace(' ', '_'), cur_word, key_rects)
    else:
        prefix_probs = None

    toks = tokenize_sofar(sofar)
    # print(repr(sofar), toks)
    next_words = beam_search_phrases(toks, beam_width=10, length=1, prefix_probs=prefix_probs, length_bonus=length_bonus)[:3]
    return toks, next_words

def predict_forward(toks, oneword_suggestion, beam_width=50, length=25, length_bonus=0):
    return dict(one_word=oneword_suggestion, continuation=beam_search_phrases(
        toks + oneword_suggestion['words'], beam_width=beam_width, length=length, length_bonus=length_bonus)[:10])


def phrases_to_suggs(phrases):
    def de_numpy(x):
        return x.tolist() if x is not None else None
    return [dict(one_word=dict(words=phrase[:1]), continuation=[dict(words=phrase[1:])], probs=de_numpy(probs)) for phrase, probs in phrases]
