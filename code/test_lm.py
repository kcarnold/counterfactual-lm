from decoder import LanguageModel, get_model, tokenize_sofar, generate_phrase


def test_score_seq():
    model = get_model('yelp_train')

    words_good = tokenize_sofar("this is a great")
    words_bad = words_good[::-1]
    score_good, _ = model.score_seq(model.bos_state, words_good)
    score_bad, _ = model.score_seq(model.bos_state, words_bad)

    assert score_good > score_bad
    assert score_good > -30

def test_next_word_logprobs():
    model = get_model('yelp_train')
    context = tokenize_sofar('')
    state, _ = model.get_state(context, bos=True)
    next_words, logprobs = model.next_word_logprobs_raw(state=state, prev_word=context[-1])

    # There is a "next_word" for each logprob.
    assert len(next_words) > 0
    assert len(next_words) == len(logprobs)

    # The "next words" include common words.
    assert 'this' in next_words
    assert 'lentil' in next_words
    assert logprobs[next_words.index('this')] > logprobs[next_words.index('lentil')]


def test_generate_phrase():
    model = get_model('yelp_train')
    context = tokenize_sofar('this ')
    phrase, logprobs = generate_phrase(model, context, 5)
    assert len(phrase) == len(logprobs) == 5
    
