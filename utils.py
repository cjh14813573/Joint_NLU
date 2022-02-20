import logging


def load_vocab(vocab_file):
    with open(vocab_file) as f:
        res = [l.strip().lower() for l in f.readlines() if len(l.strip()) != 0]
    return res, dict(zip(res, range(len(res)))), dict(zip(range(len(res)),res))

def get_ckpt_filename(name, epoch):
    return '{}-{}.ckpt'.format(name,epoch)

def get_logger(filename, print2screen=True):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line:%(lineno)d][%(levelname)s] \
                                  >> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    if print2screen:
        logger.addHandler(ch)
    return logger