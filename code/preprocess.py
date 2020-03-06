import numpy as np
from scipy.special import logsumexp
import kenlm
import dateutil.parser
from collections import defaultdict
import traceback
import json
import tqdm


def jsonl(fname):
    return (json.loads(line) for line in open(fname))


def parse_logs(logs):
    first_timestamp = dateutil.parser.parse(logs[0]['timestamp'])
    by_page = defaultdict(list)
    cur_state = {}
    for entry in logs:
        entry = entry.copy()
        entry['time'] = (dateutil.parser.parse(entry.pop('timestamp')) - first_timestamp).total_seconds()
        cur_state.update(entry.get('set_state', {}))
        cur_page = cur_state['page']
        if cur_page in ['tutorial', 'experiment']:
            cur_page = '{}_{}'.format(cur_page, cur_state['idx'])
            entry['condition'] = cur_state['condition']
            by_page[cur_page].append(entry)
    return dict(by_page)


def get_words(sug):
    words = sug['one_word']['words'][:]
    if len(sug['continuation']):
        words.extend(sug['continuation'][0]['words'])
    return words


def get_final_text(log):
    for line in reversed(log):
        if line.get('name') == 'updateCompose':
            return line['curText']

def track_suggestions(log):
    '''
    Returns a list of suggestions and what happened to them.

    [{'context', 'words', 'num_accepted_words', 'appeared_at', 'visible_secs'}]
    '''
    first_timestamp = dateutil.parser.parse(log[0]['timestamp'])
    final_text = get_final_text(log)
    suggs = []
    cur_suggs = [None] * 4
    cur_text = ''
    for entry_idx, entry in enumerate(log):
        typ = entry.get('name')
        now = (dateutil.parser.parse(entry['timestamp']) - first_timestamp).total_seconds()
        if typ == 'updateCompose':
            cur_text = entry['curText']
        elif typ == 'showSuggestions':
            new_suggs = entry['suggestions']['next_word']
            for i in range(4):
                sug = new_suggs[i] if i < len(new_suggs) else None
                if cur_suggs[i] is not None:
                    old_sugg = cur_suggs[i]
                    # See if this was a phrase suggestion continuation
                    if sug is not None and get_words(sug)[:1] == old_sugg['words'][old_sugg['num_accepted_words']:][:1]:
                        # Don't overwrite.
                        continue
                    old_sugg['visible_secs'] = now - old_sugg['appeared_at']
                    suggs.append(old_sugg)
                    cur_suggs[i] = None
                if sug is None:
                    continue
                cur_suggs[i] = dict(pid=entry['participant_id'], context=cur_text, words=get_words(sug), num_accepted_words=0, slot=i, appeared_at=now, visible_secs=None)
        elif typ == 'insertedSuggestion':
            prev_sugg = cur_suggs[entry['slot']]
            if prev_sugg is None or prev_sugg['words'][prev_sugg['num_accepted_words']] !=entry['toInsert']:
                print("alert: mismatched toInsert", entry['toInsert'])
#                assert False
            prev_sugg['num_accepted_words'] += 1
    for i in range(3):
        sug = cur_suggs[i]
        if sug is not None:
            sug['visible_secs'] = now - sug['appeared_at']
            suggs.append(sug)
    return [sugg for sugg in suggs if final_text.startswith(sugg['context']) and (sugg['num_accepted_words'] > 0 or sugg['visible_secs'] > .3)]

all_logs = {}
import glob
for fn in glob.glob('../dataset/logs/*.jsonl'):
    log = list(jsonl(fn))
    all_logs[log[0]['participant_id']] = log
suggs = {part_id: track_suggestions(log) for part_id, log in all_logs.items()}
final_reviews = {part_id: get_final_text(log) for part_id, log in all_logs.items()}

import re
num_words = [len(re.findall(r'\w+', doc)) for part_id, doc in sorted(final_reviews.items())]

print()
print("{} participants".format(len(suggs)))
print('word count: mean={:.1f} std={:.2f}'.format(np.mean(num_words), np.std(num_words)))

flat_suggs = [s for sugg_list in suggs.values() for s in sugg_list]
accepted_word_counts = np.bincount([s['num_accepted_words'] for s in flat_suggs])

import pandas as pd
pd.DataFrame(flat_suggs).num_accepted_words.value_counts()

# Exclude one participant who typed the review entirely by tapping suggestions.
for excl in ['597763']:
    if excl in suggs:
        suggs.pop(excl)

# Unfortunately we forgot to log the generation probability when making the suggestions.
# Reconstruct it by running the suggestion generator again...

import decoder
model = decoder.get_model('yelp_train')

def tokenize_sofar(sofar):
    toks = decoder.tokenize_mid_document(sofar.lower().replace(' .', '.').replace(' ,', ','))[0]
    assert toks[0] == "<D>"
    assert toks[1] == "<P>"
    assert toks[2] == "<S>"
    cur_word = toks[-1]
    toks = ['<s>', "<D>"] + toks[3:-1]
    return toks, cur_word

def score_sugg(context, words, temperature=.5):
    context_toks, cur_word = tokenize_sofar(context)
    context_toks = context_toks[-6:]
    prefix_logprobs = [(0., cur_word)] if len(cur_word) else None
    state, _ = model.get_state(context_toks)
    scores = []
    for i, word in enumerate(words):
        word_idx = model.model.vocab_index(word)
        next_words, logprobs = model.next_word_logprobs_raw(state, context_toks[-1] if i == 0 else words[i-1], prefix_logprobs=prefix_logprobs)
        prefix_logprobs = None
        logprobs /= temperature
        scores.append(logprobs[next_words.index(word_idx)] - logsumexp(logprobs))

        # Advance the model state.
        new_state = kenlm.State()
        model.model.base_score_from_idx(state, word_idx, new_state)
        state = new_state
    return scores

for sugg in tqdm.tqdm(flat_suggs):
    scores = None
    try:
        scores = score_sugg(sugg['context'], sugg['words'])
    except Exception:
        traceback.print_exc()
        print(sugg)
    sugg['generation_probs'] = scores

df = pd.DataFrame([dict(s, words=' '.join(s['words'])) for s in flat_suggs])
del df['appeared_at']
df.rename(columns=dict(pid='participant_id')).to_csv('analyzed/by_suggestion.csv', index=False)

fr = pd.Series(final_reviews)
fr.index.name = 'participant_id'
fr.to_frame('final_review').to_csv('analyzed/final_reviews.csv')

if False:

    by_context = defaultdict(list)
    for sugg in flat_suggs:
        if sugg['generation_prob'] is None:
            continue
        if len(sugg['context']) > 0 and not sugg['context'].endswith(' '):
            continue
        by_context[sugg['context']].append(sugg)
    #%%
    at_bos = [s for s in flat_suggs if s['generation_prob'] is not None]
    at_bos = [s for s in at_bos if len(s['context']) == 0 or s['context'].endswith(' ')]
    at_bos = [s for s in at_bos if tokenize_sofar(s['context'])[0][-1] in ['<S>', '<D>']]
    #%%
    at_bos_accepted = [s for s in at_bos if s['num_accepted_words'] > 2]
    #%%
    by_context_any_accepted = {context: suggs for context, suggs in by_context.items() if any(s['num_accepted_words'] > 1 for s in suggs) and len(suggs) == 3}
    #%%
    gen_spread = {context: max([s['generation_prob'] for s in ss]) - min([s['generation_prob'] for s in ss]) for context, ss in by_context_any_accepted.items()}
    #%%
    examples = []
    for context, suggs in by_context_any_accepted.items():
        probs = [s['generation_prob'] for s in suggs]
        which_accepted = [i for i, s in enumerate(suggs) if s['num_accepted_words']][0]
        diff = max(prob for i, prob in enumerate(probs) if i != which_accepted) - probs[which_accepted]
        if diff > 0:
            examples.append((diff, context))

    for diff, context in sorted(examples)[-5:]:
        print()
        print(context)
        for s in by_context[context]:
            print(' {} {:.2f} {}'.format(s['num_accepted_words'], s['generation_prob'], ' '.join(s['words'])))

    #%%
    context = "chipotle is one of my favorite restaurants. the food here is pretty good, and the portions were just right. the menu is very diverse and there is always something to get that you'll enjoy. the service here was great! "
    for option in ['i have to', 'the food is', 'they are always']:
        print(option, score_sugg(context, option.split()))
    #%%
    context = "the food was really tasty. it was definitely a "
    for option in ['plus . and the place is', 'good place to go to .', 'nice change from a day']:
        option = ' '.join(option.split())
        scores = score_sugg(context, option.split())
        print(option, np.round(np.cumsum(scores), 2).tolist())
    by_context[context]
    #%%
    by_context["so i went to chipotle last night with my boyfriend and we got the burrito bowls with chicken and veggies, and everything was delicious. the food was fast and "]
