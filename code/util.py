import os
from contextlib import contextmanager

def save_fig(basename, figpath='.'):
    import datetime
    filename = 'figures/{}-{}.pdf'.format(basename, datetime.datetime.now().isoformat().replace(':', '-'))
    import matplotlib.pyplot as plt
    if not os.path.isabs(filename):
        filename = os.path.join(figpath, filename)
    plt.savefig(filename+'.pdf')


@contextmanager
def fig(basename, **kw):
    import matplotlib.pyplot as plt
    fig = plt.figure(**kw)
    yield fig
    save_fig(basename)
    plt.close(fig)


import numpy as np
from scipy.special import logsumexp
try:
    import numba
    @numba.jit(nopython=True)
    def fast_logsumexp(a):
        a_max = np.max(a)
        out = a - a_max
        np.exp(out, out)
        return np.log(np.sum(out)) + a_max
    logsumexp = fast_logsumexp
except:
    print("Falling back to slow logsumexp")
