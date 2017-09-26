import json
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import util

sns.set_style('whitegrid')

#%% Load model
import decoder
all_model = decoder.get_model('yelp_train')
all_model.prune_bigrams()

from counterfactual import get_all_features, bounded_estimate_reward, Objective, contextual_expected_reward_samples

# NOTE: temperature is the *inverse* of weights[0].
ref_weights= np.r_[2., 0., np.zeros(len(all_model.id2tag))]
NUM_WEIGHTS = len(ref_weights)



#%%
num_vocab_words = len(all_model.vocab_trie)
def partial_word_score(prefix):
    '''Return the amount of credit to give accepting a word starting with prefix.'''
    if prefix == '':
        return 1.
    return len(all_model.vocab_trie.keys(prefix)) / num_vocab_words
partial_word_score(''), partial_word_score('as')
#%%
def tokenize_sofar(sofar):
    toks = decoder.tokenize_mid_document(sofar.lower().replace(' .', '.').replace(' ,', ','))[0]
    assert toks[0] == "<D>"
    assert toks[1] == "<P>"
    assert toks[2] == "<S>"
    cur_word = toks[-1]
    toks = ['<s>', "<D>"] + toks[3:-1]
    return toks, cur_word
#%%
real_suggs = pd.read_csv('analyzed/by_suggestion.csv')
real_suggs['generation_probs'] = real_suggs.generation_probs.map(lambda x: json.loads(x) if isinstance(x, str) else None)

#%% Transform suggestions into (context, actions-taken, chosen, reward) tuples.
contexts = []
contexts_tok = []
suggestions = []
chosen = []
rewards = []
num_contexts = 0
num_suggestions_offered = 0

# NOTE: This is a tad wonky because a context will appear multiple times if
# the author backspaces. Some logging / preprocessing bugs might also
# introduce some noise here.
for (part_id, context), data in real_suggs.groupby(('participant_id', 'context')):
    num_contexts += 1
    num_suggs = len(data)
    suggs = [(x.words.split(), x.generation_probs) for x in data.itertuples()]
    which_chosen = np.argmax(data.num_accepted_words.data)
    reward = data.num_accepted_words.iloc[which_chosen]
    toks, cur_word = tokenize_sofar(context)
    if cur_word:
        reward = reward - 1 + partial_word_score(cur_word)
    if reward < 1:
        # Mid-word acceptance that wasn't followed by accepting more words of the suggestion.
        # Assume the user was just completing the current word.
        # Give us neither credit nor blame for this "suggestion".
        continue
    # At this point, consider that the suggestion was, verily, offered.
    num_suggestions_offered += num_suggs
    # if reward == 0:
    #     # Suggestion actually not accepted.
    #     continue
    contexts.append(context)
    contexts_tok.append((toks, cur_word))
    suggestions.append(suggs)
    chosen.append(which_chosen)
    rewards.append(reward)
#%%
print('Accepted {:.1%} of suggestions, {} words, avg {:.2f} words per sugg offered'.format(
      len(contexts) / num_suggestions_offered,
      np.sum(rewards),
      np.sum(rewards) / num_suggestions_offered))
num_suggs = np.array([len(suggs) for suggs in suggestions])
np.mean(num_suggs == 2)
#%%
#%%
all_features = get_all_features(contexts_tok, suggestions, base_model=all_model)
#%%
is_valid = np.array([len(suggs) > 1 and all(x is not None for x in suggs) for suggs in all_features])
print("Tossing", 1-np.mean(is_valid))
contexts_tok = np.array(contexts_tok)[is_valid]
suggestions = np.array(suggestions)[is_valid]
features = np.array(all_features)[is_valid]
chosen = np.array(chosen)[is_valid]
rewards = np.array(rewards)[is_valid]
#%%
generation_logprobs = np.array([sugg[chosen][1] for sugg, chosen in zip(suggestions, chosen)])
features_chosen = np.array([feat[cho] for feat, cho in zip(features, chosen)])
#%%
def with_special_features_cleared(feat):
    features_concat = feat['features_concat']
    features_concat = [f[:, :1] for f in features_concat]
    return dict(feat, features_concat=features_concat)

features_chosen_justtemp = np.array([with_special_features_cleared(feat) for feat in features_chosen])


#%%
M = 10.
def estimate_reward(cers, num_offered, M=M):
    return bounded_estimate_reward(cers, num_suggestions_offered=num_offered, M=M, delta=.95)

#%%
baselines = {}
baselines['ref'] = ref_weights.copy()

cers_baselines = {k: contextual_expected_reward_samples(weights, suggestions, features, chosen, rewards) for k, weights in baselines.items()}
#%%
#frac_not_accepted = len(contexts_tok) / num_contexts
#%%
K_fold = 5
fold_ids = np.array([i % K_fold for i in range(len(contexts_tok))])
fold_results = []

print('Orig: avg {:.2f} words per sugg offered'.format(
      np.sum(rewards) / num_suggestions_offered))
for fold_id in range(K_fold):
    train = fold_ids != fold_id
    test = fold_ids == fold_id
    effective_train_contexts = np.sum(train) / len(contexts_tok) * num_contexts
    effective_test_contexts = np.sum(test) / len(contexts_tok) * num_contexts
    print("Fold", fold_id, np.sum(train), np.sum(test))

    datum_train = {k: estimate_reward(cers[train], num_offered=num_suggestions_offered) for k, cers in cers_baselines.items()}
    datum_test = {k: estimate_reward(cers[test], num_offered=num_suggestions_offered) for k, cers in cers_baselines.items()}

    rs = np.random.RandomState(0)
    obj = Objective(num_suggestions_offered, features_chosen_justtemp[train], generation_logprobs[train], rewards[train], M=10, regularization=0.)
    x0 = np.array([1.0])
    res = minimize(obj, x0, jac=True, options=dict(disp=True))
    assert res.x[0] > 0
    if res.nit < 2:
        print("WARNING: SMALL # OF ITERATIONS", repr(res.x))
    justtemp_weights = np.r_[res.x, np.zeros(NUM_WEIGHTS-1)]
    print("Justtemp objective {:.2f}".format(
          obj(res.x)[0]))
    cers_justtemp_all = contextual_expected_reward_samples(justtemp_weights, suggestions, features, chosen, rewards)
    datum_train['justtemp'] = estimate_reward(cers_justtemp_all[train], num_offered=num_suggestions_offered)
    datum_test['justtemp'] = estimate_reward(cers_justtemp_all[test], num_offered=num_suggestions_offered)

    rs = np.random.RandomState(0)
    obj = Objective(num_suggestions_offered, features_chosen[train], generation_logprobs[train], rewards[train], M=10, regularization=0.)
    while True:
        x0 = rs.standard_normal(NUM_WEIGHTS)
        x0[0] = 1.
        res = minimize(obj, x0, jac=True, options=dict(disp=True))
        assert res.x[0] > 0
        if res.nit < 2:
            print("WARNING: SMALL # OF ITERATIONS, retry", repr(res.x))
            continue
        break
    weights = res.x
    print("Optimized objective {:.2f}".format(
          obj(res.x)[0]))
    cers_opt_all = contextual_expected_reward_samples(weights, suggestions, features, chosen, rewards)
    datum_train['opt'] = estimate_reward(cers_opt_all[train], num_offered=num_suggestions_offered)
    datum_test['opt'] = estimate_reward(cers_opt_all[test], num_offered=num_suggestions_offered)
    print('Train:', ', '.join('{}: {:.2f}'.format(name, val) for name, val in sorted(datum_train.items())))
    print('Test: ', ', '.join('{}: {:.2f}'.format(name, val) for name, val in sorted(datum_test.items())))
    print('\n')
    # data_of_interest.append((datum_train, datum_test))
    fold_results.append(((weights, cers_opt_all), (justtemp_weights, cers_justtemp_all)))

#%%
orig_reward = np.sum(rewards) / num_suggestions_offered
print("Orig: {:.2f}".format(orig_reward))
train_data = []
test_data = []
for fold_id, ((weights, cers_opt_all), (justtemp_weights, cers_justtemp_all)) in enumerate(fold_results):
    train = fold_ids != fold_id
    test = fold_ids == fold_id
    num_train = np.sum(train)
    num_test = np.sum(test)
    for M in np.r_[1:100:2]:
        datum_train = {k: estimate_reward(cers[train], num_train, M=M) for k, cers in cers_baselines.items()}
        datum_test = {k: estimate_reward(cers[test], num_test, M=M) for k, cers in cers_baselines.items()}
        datum_train['opt'] = estimate_reward(cers_opt_all[train], num_train, M=M)
        datum_test['opt'] = estimate_reward(cers_opt_all[test], num_test, M=M)
        datum_train['justtemp'] = estimate_reward(cers_justtemp_all[train], num_train, M=M)
        datum_test['justtemp'] = estimate_reward(cers_justtemp_all[test], num_test, M=M)
        datum_train['M'] = M
        datum_test['M'] = M
#    print('Train:', ', '.join('{}: {:.2f}'.format(name, val) for name, val in sorted(datum_train.items())))
#    print('Test: ', ', '.join('{}: {:.2f}'.format(name, val) for name, val in sorted(datum_test.items())))
#    print('\n')
        train_data.append(datum_train)
        test_data.append(datum_test)
#%%
df = pd.DataFrame(test_data)
#sns.tsplot(df, time='M', value='opt)
#%%
with util.fig('estimated improvement varying M') as f:
    df.rename(columns=dict(justtemp="Best reweighting of base LM", opt="Fully adapted model", ref="Logging policy")).groupby('M').mean().plot()
    plt.ylabel("Estimated reward (words accepted per suggestion offered)")
    plt.xlabel("Truncation factor M used in estimation")
#%%
tag_pretty = {'.': 'PUNCT'}
id2tag_read = [tag_pretty.get(tag, tag) for tag in all_model.id2tag]
weights_df = pd.DataFrame(np.array([weights for (weights, cers), (justtemp_weights, cers_justtemp_all) in fold_results]), columns=['log_likelihood', 'is_long'] + id2tag_read)
print(pd.DataFrame(dict(mean=weights_df.mean(), std=weights_df.std())).T.to_latex(float_format='{:.2f}'.format))
#print(' & '.join('{:.2f} \pm ))
#weights_df..describe().T

#%% Dump some examples.
mean_cers_opt = np.mean([f[0][1] for f in fold_results], axis=0)
log_importance_ratios = np.array([p - q for q, p, reward in mean_cers_opt])
rewards = np.array([reward for q, p, reward in mean_cers_opt])
estimated_reward_samples = rewards * np.minimum(M, np.exp(log_importance_ratios))

suggestions_chosen = [' '.join(suggestions[i][chosen[i]][0]) for i in range(len(suggestions))]

print("Top suggestions (context, suggestion, # words accepted)")
for i in np.argsort((log_importance_ratios))[-10:]:
    q, p, reward = mean_cers_opt[i]
    probs_to_use = int(np.ceil(reward+1e-6))
    print(f'Context & {" ".join(contexts_tok[i][0][-5:])}')
    print(f'Suggestion & {" ".join(suggestions[i][chosen[i]][0][:probs_to_use])}')
    print(f"Reward & {rewards[i]:.2f}")
    print(f"Logprobs & orig: {q:.2f}, new: {p:.2f}")
    # print("Importance ratio", p - q)
    print()

print("\n\nBottom suggestions (context, suggestion, # words accepted)")
for i in np.argsort((log_importance_ratios))[:10]:
    q, p, reward = mean_cers_opt[i]
    probs_to_use = int(np.ceil(reward+1e-6))
    print(f'Context & {" ".join(contexts_tok[i][0][-5:])}')
    print(f'Suggestion & {" ".join(suggestions[i][chosen[i]][0][:probs_to_use])}')
    print(f"Reward & {rewards[i]:.2f}")
    print(f"Logprobs & orig: {q:.2f}, new: {p:.2f}")
    # print("Importance ratio", p - q)
    print()


#%%
with util.fig("generation_probs_for_accepted_suggestions") as f:
    sns.kdeplot(mean_cers_opt[:,0], label="generation")
    sns.kdeplot(mean_cers_opt[:,1], label="learned")

#sns.kdeplot(mean_cers_opt[:,1]-mean_cers_opt[:,0])

with util.fig("cumulative_rewards") as f:
    mco_as = np.argsort(mean_cers_opt[:,0]); plt.plot(mean_cers_opt[:,0][mco_as], np.cumsum(mean_cers_opt[:,2][mco_as]), label="Logging policy")
    mco_as = np.argsort(mean_cers_opt[:,1]); plt.plot(mean_cers_opt[:,1][mco_as], np.cumsum(mean_cers_opt[:,2][mco_as]), label="Learned policy", color='g')
    plt.xlabel('Generation likelihood')
    plt.ylabel("Cumulative reward")
    plt.legend(loc='best')
