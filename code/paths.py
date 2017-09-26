import os


class paths:
    parent = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    ui = os.path.join(parent, 'ui')
    logdir = os.path.join(parent, 'logs')
    db = os.path.join(parent, 'database.db')

    cache = os.path.join(parent, 'cache')

    @classmethod
    def model_basename(cls, model_name):
        return os.path.join(cls.parent, 'models', model_name)
