import pickle
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import string
import pandas as pd
import seaborn as sns
import tqdm
from joblib import Memory
import util

from util import logsumexp
from counterfactual import get_features, Objective, contextual_expected_reward_samples


#%%
mem = Memory('cache')

#%% Load docs
print("Loading docs")
docsets = {k: pickle.load(open('../yelp_preproc/{}_data.pkl'.format(k), 'rb')) for k in ['train', 'valid', 'test']}

#%% Load model
import decoder
from paths import paths
all_model = decoder.Model.from_basename(paths.model_basename('yelp_all_as_sents'))
all_model.prune_bigrams()

#%%
num_samples = 2000
num_suggestions_per_context = 3
num_words_per_suggestion = 5
num_suggestions_offered = num_samples * num_suggestions_per_context

#%%
@mem.cache()
def gen_suggs(num_contexts, docset, rand_seed, **params):
    params.setdefault('temperature', .5)

    docs = docsets[docset]
    rs = np.random.RandomState(rand_seed)
    np.random.seed(rand_seed)
    sampled_docids = rs.choice(len(docs), num_contexts, replace=False)
    sampled_texts = docs.tokenized.iloc[sampled_docids]
    contexts = []
    suggestions = []
    for text in tqdm.tqdm(sampled_texts):
        while True:
            sent = rs.choice(text.split('\n')).lower().split()
            loc = rs.choice(len(sent))
            context = sent[:loc]
            try:
                suggs = decoder.generate_diverse_phrases(all_model, ['<s>'] + context,
                                                         num_suggestions_per_context,
                                                         num_words_per_suggestion, **params)
                if len(suggs) < 3:
                    continue
            except decoder.GenerationFailedException:
                continue
            contexts.append(context)
            suggestions.append(suggs)
            break
    return contexts, suggestions

#%%
print("Generate initial suggestions")
contexts, ref_suggestions = gen_suggs(num_contexts=num_samples, docset='train', rand_seed=0)

#%% Define some desirability functions
def desire_null(context, phrase):
    return 0.

def desire_longword(context, phrase, length_bonus_min_length=6):
    word_lengths = np.array([len(w) >= length_bonus_min_length if w[0] in string.ascii_letters else 0 for w in phrase])
    return np.mean(word_lengths)


#%%
def log_left_over(xx):
    return np.log1p(-np.sum(np.exp(xx)))

#%% Simulate acceptance behavior
# Define actions
actions = [(None, 0)] + [(sug_no, num_words) for sug_no in range(num_suggestions_per_context) for num_words in range(1, num_words_per_suggestion + 1)]
act_which_sugg, act_num_words = zip(*actions)
act_which_sugg = np.array(act_which_sugg)
act_num_words = np.array(act_num_words)
#%%

from collections import namedtuple
class Policy(namedtuple('Policy', 'desire gain bias after_first_word_scaling')):
    @classmethod
    def make(cls, desire, gain=1., bias=0., after_first_word_scaling=1.):
        return cls(desire, gain, bias, after_first_word_scaling)


def action_scores(context, suggs, policy, base_model=all_model):
    '''Compute logits for possible actions for a batch of suggestions.

    suggs: [(words, gen_logprobs)]
    desirability_fn: words -> [desirability(accept up to word i) for i in 1..|W|]
    '''
    desirability_fn = policy.desire
    gain = policy.gain
    bias = policy.bias
    after_first_word_scaling = policy.after_first_word_scaling

    state = base_model.get_state(context, bos=True)[0]
    sugg_words = [words for words, _ in suggs]
    assert all(len(words) == num_words_per_suggestion for words in sugg_words)
    predictive_logprobs = [base_model.score_seq_by_word(state, words) for words in sugg_words]
    # Desirability is per _suggestion_
    desirabilities = gain * np.array([desirability_fn(context, words) for words in sugg_words]) + bias
    assert len(desirabilities) == num_suggestions_per_context

    # Pick a suggestion, or null, according to the desirability-biased probability of its first word.
    raw_logprobs = [plp[0] for plp in predictive_logprobs]
    assert len(raw_logprobs) == num_suggestions_per_context
    everything_else_logprob = log_left_over(raw_logprobs)
    fw_option_logprobs = np.r_[everything_else_logprob, raw_logprobs + desirabilities]
    fw_option_logprobs -= logsumexp(fw_option_logprobs)

    # Continue it according to the conditional distribution.
    logprobs_by_action = [fw_option_logprobs[0]] # null action
    for sug_idx, (baseline_logprob, words, plp, desired) in enumerate(zip(fw_option_logprobs[1:], sugg_words, predictive_logprobs, desirabilities)):
        # Conditional on this suggestion being chosen, what's the prob of picking n words?
        # Consider stopping at each word in succession
        for word_idx, word in enumerate(words[1:], start=1):
            logprob = plp[word_idx]
            everything_else_logprob = np.log1p(-np.exp(logprob))
            option_logprobs = np.r_[everything_else_logprob, logprob + desired * after_first_word_scaling]
            option_logprobs -= logsumexp(option_logprobs)
            option_logprobs += baseline_logprob
            # The probability of accepting n words is the probability of stopping at word n+1
            logprobs_by_action.append(option_logprobs[0])
            baseline_logprob = option_logprobs[1]
        logprobs_by_action.append(baseline_logprob)
    assert len(logprobs_by_action) == len(actions)
    assert np.isclose(logsumexp(np.array(logprobs_by_action)), 0.)
    return logprobs_by_action


def all_action_scores(contexts, suggestions, policy, base_model=all_model):
    return np.array([action_scores(context, suggs, policy, base_model=base_model)
        for context, suggs in zip(contexts, suggestions)])

null_policy = Policy.make(desire_null)
null_scores = all_action_scores(contexts, ref_suggestions, null_policy)

longword_policy = Policy.make(desire_longword, gain=10)
longword_scores = all_action_scores(contexts, ref_suggestions, longword_policy)

#%%
print("The two policies agree on which suggestion to take {:.1%}".format(
    np.mean(act_which_sugg[np.argmax(null_scores, axis=1)] ==
            act_which_sugg[np.argmax(longword_scores, axis=1)])))
#%%
def summarize_policy(name, logprobs):
    probs = np.exp(logprobs)
    expected_numwords = np.sum(act_num_words * probs, axis=1)
    print("{} would accept {:.1%} +- {:.1%}, expect {:.1f} words per suggestion +- {:.2f}".format(
          name,
          np.sum(1-probs[:, 0]) / num_suggestions_offered,
          np.std(1 - probs[:, 0]) / np.sqrt(num_suggestions_offered),
          np.sum(expected_numwords)/num_suggestions_offered,
          np.std(expected_numwords) / np.sqrt(num_suggestions_offered)))
summarize_policy('null', null_scores)
summarize_policy('longword', longword_scores)


#%%
def get_all_features(contexts, suggestions, base_model=all_model):
    return [[get_features(base_model, context, sugg_words) for sugg_words, _ in suggs] for context, suggs in zip(tqdm.tqdm(contexts), suggestions)]

print("Computing features for ref suggestions")
ref_feats = get_all_features(contexts, ref_suggestions)

#%%
def collect_acceptances(suggestions, features, action_choices, base_model=all_model):
    suggss = []
    all_features = []
    accepted_features = []
    generation_logprobs = []
    chosens = []
    observed_rewards = []
    for suggs, feats, choice in zip(suggestions, features, action_choices):
        if choice == 0:
            continue
        sug_no, num_words = actions[choice]
        sugg_words, gen_probs = suggs[sug_no]
        generation_logprobs.append(gen_probs)
        suggss.append(suggs)
        all_features.append(feats)
        accepted_features.append(feats[sug_no])
        chosens.append(sug_no)
        observed_rewards.append(num_words)
    return suggss, all_features, accepted_features, generation_logprobs, chosens, observed_rewards

#%%
rs = np.random.RandomState(100)
longword_action_choices = np.array([rs.choice(len(p), p=np.exp(p)) for p in longword_scores])
print('Accepted {:.1%} of suggestions, {} words, avg {:.2f} words per sugg offered'.format(
      np.sum(longword_action_choices > 0) / num_suggestions_offered,
      np.sum(act_num_words[longword_action_choices]),
      np.sum(act_num_words[longword_action_choices]) / num_suggestions_offered))
#%%
suggss, all_features, features_chosen, generation_logprobs, chosens, observed_rewards = collect_acceptances(
    ref_suggestions, ref_feats, longword_action_choices)

#%%
#xx = np.random.standard_normal(2000)
#samps = xx[np.random.randint(len(xx), size=(len(xx), 10000))]
#means = np.mean(samps, axis=0)
##low_percentile = np.percentile(samps, 2.5, axis=0)
#sns.distplot(means)
#(1/np.std(means))**2
#%%

#%% Compute TIP-estimated expected reward given a dataset of suggestion -> acceptances

# NOTE: temperature is the *inverse* of weights[0].
ref_weights= np.r_[2., 0., np.zeros(12)]
NUM_WEIGHTS = len(ref_weights)

#%%
def plot_expected_reward_estimate(cers, Ms, orig_scores, new_scores, filt=None):
    qs, ps, rewards = np.array(cers).T
    importance_ratio = np.exp(ps-qs)
    estimated_reward = rewards[:,None] * np.minimum(importance_ratio[:,None], Ms)
    if filt is not None:
        estimated_reward = estimated_reward[filt]
    frac_accepted = len(estimated_reward) / len(orig_scores)
    estimated_reward *= frac_accepted
    mean = np.mean(estimated_reward, axis=0)
    sem = .1*Ms
#    sem = np.std(estimated_reward, axis=0) / np.sqrt(len(ps))
    plt.fill_between(Ms, mean - sem, mean + sem, color=[.8,.8,.8])
    plt.plot(Ms, mean)
#    plt.errorbar(Ms, [np.mean((rewards * np.minimum(M, np.exp(ps-qs)))) for M in Ms],
#                      yerr=[2*np.std(rewards * np.minimum(M, np.exp(ps-qs)))/np.sqrt(len(ps)) for M in Ms])
#    plt.plot(Ms, [np.mean((rewards * np.minimum(M, np.exp(ps-qs)))) - 2*np.std(rewards * np.minimum(M, np.exp(ps-qs)))/np.sqrt(len(ps)) for M in Ms])
    if orig_scores is not None:
        original_expected = np.mean(np.sum(act_num_words * np.exp(orig_scores), axis=1))
        plt.hlines(original_expected, Ms.min(), Ms.max(), color='gray')
    if new_scores is not None:
        new_expected = np.mean(np.sum(act_num_words * np.exp(new_scores), axis=1))
        plt.hlines(new_expected, Ms.min(), Ms.max())
    plt.xlabel('M')
    plt.ylabel('estimated expected reward, by context')

#%% Generate a new set of suggestions under the new suggestion policy
def weights_to_model_params(weights):
    return dict(
        temperature=1./weights[0],
        length_bonus_amt=weights[1],
        pos_weights=weights[2:])

#%%
def actual_expected_reward_from_weights(weights, accept_policy, num_contexts=500):
    model_params = weights_to_model_params(weights)
    contexts, suggestions = gen_suggs(num_contexts=num_contexts, docset='valid', rand_seed=0, **model_params)
    scores_under_policy = all_action_scores(contexts, suggestions, accept_policy)
    expected_num_words = np.sum(act_num_words * np.exp(scores_under_policy), axis=1)
    return np.mean(expected_num_words), np.std(expected_num_words) / np.sqrt(num_contexts)

#%% Vary amount of data and M, train model, compute actual expected reward under new suggestion policy.
def subsample_trial(i, num_train, M, ref_suggestions, ref_feats, ref_scores, accept_policy, val_num_contexts=500):
    print("Starting trial", i, num_train, M)
    # Subsample.
    assert num_train <= len(ref_suggestions)
    rs = np.random.RandomState(i)
    train_samples = rs.choice(len(ref_suggestions), num_train, replace=False)
    ref_suggestions = [ref_suggestions[i] for i in train_samples]
    ref_feats = [ref_feats[i] for i in train_samples]
    ref_scores = [ref_scores[i] for i in train_samples]

    # Pick actions.
    action_choices = np.array([rs.choice(len(p), p=np.exp(p)) for p in ref_scores])
    suggs, features, features_chosen, generation_logprobs, chosens, observed_rewards = collect_acceptances(
        ref_suggestions, ref_feats, action_choices)

    # Fit weights
    obj = Objective(num_suggestions_offered, features_chosen, generation_logprobs, observed_rewards, M=M, regularization=0.)
    x0 = ref_weights
    while True:
        res = minimize(obj, x0, jac=True, options=dict(disp=True))
        assert res.x[0] > 0
        if res.nit < 2:
            print("WARNING: SMALL # OF ITERATIONS, retry", i, num_train, M, repr(res.x))
            x0 = rs.standard_normal(NUM_WEIGHTS)
            x0[0] = 1.
            continue
        break
    weights = res.x

    # Expected reward
    cers = contextual_expected_reward_samples(weights, suggs, features, chosens, observed_rewards)
    # Evaluate actual reward
    mean, sem = actual_expected_reward_from_weights(weights, accept_policy, num_contexts=val_num_contexts)

    return weights, cers, mean, sem

#%%
def trial_wrapper(args):
    import traceback
    i, num_train, M, scores, policy = args
    try:
        return subsample_trial(i=i, num_train=num_train, ref_scores=scores, M=M, ref_suggestions=ref_suggestions, ref_feats=ref_feats, accept_policy=policy)
    except Exception:
        traceback.print_exc()

#%%
from joblib import Parallel, delayed
def run_subsample_trial(basename, num_trains, Ms, policy):
    scores = all_action_scores(contexts, ref_suggestions, policy)
    params = [(num_train, M) for num_train in num_trains for M in Ms]
    trials = Parallel(n_jobs=-1, verbose=10, backend='multiprocessing')(
        delayed(trial_wrapper)((i, num_train, M, scores, policy))
        for i, (num_train, M) in enumerate(params))
    with open(basename+'-subsample_trials.pkl', 'wb') as f:
        pickle.dump(dict(params=params, trials=trials, baseline_scores=scores), f, -1)

NAME = 'longword5'
run_subsample_trial(
    NAME,
    # num_trains=np.logspace(0, np.log10(1000),10).astype(int),#np.logspace(2, np.log10(2000), 5).astype(int),
    num_trains=np.linspace(10, 2000, 20).astype(int),
    Ms=np.ones(1) * 10.,
    policy=longword_policy)

#%%
def bounded_estimate_reward(cers, orig_num_suggestions, M):
    qs, ps, rewards = np.array(cers).T
    importance_ratio = np.exp(ps-qs)
    return np.sum(rewards * np.minimum(importance_ratio, M)) / orig_num_suggestions

#%%
def plot_results(data, name):
    subsample_trial_params = data['params']
    subsample_trials = data['trials']
    baseline_scores = data['baseline_scores']
    processed_data = []
    for params, trial_data in zip(subsample_trial_params, subsample_trials):
        if trial_data is None:
            continue
        num_train, M = params
        weights, cers, mean, sem = trial_data
        processed_data.append(dict(
            num_train=num_train, M=M,
            weights=weights, cers=cers, mean=mean, sem=sem,
            reward_estimate=bounded_estimate_reward(cers, M=M, orig_num_suggestions=num_train)))
    df = pd.DataFrame(processed_data)
    with util.fig(f"{name}_simulation_reward_by_training_set_size"):
        sns.set_style('whitegrid')
        for color, (M, data) in zip(sns.color_palette(), df.groupby('M')):
            if M < 3: continue
            data_mean = data.groupby('num_train').mean().reset_index()
            data_sem = data.groupby('num_train').sem().reset_index()
            plt.plot(data_mean['num_train'], data_mean['reward_estimate'], ':', label='counterfactual estimate from training'.format(M), color=color)
            plt.plot(data_mean['num_train'], data_mean['mean'], label='actual performance on testing'.format(M), color=color)
        orig_reward = np.mean(np.sum(act_num_words * np.exp(baseline_scores), axis=1))
        plt.hlines(orig_reward, *plt.xlim(), label='logging policy $h_0$')
        plt.legend(loc='best')
        plt.xlabel("# training samples")
        plt.ylabel("Reward (# words per suggestion)")
        plt.ylim([0, 4])

d = pickle.load(open(f'{NAME}-subsample_trials.pkl', 'rb'))
plot_results(d, NAME)

#%% Try varying: individual variation in propensity to accept, varying M, variyng amount of data

