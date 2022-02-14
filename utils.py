

def load_vocab(vocab_file):
    with open(vocab_file) as f:
        res = [l.strip().lower() for l in f.readlines() if len(l.strip()) != 0]
    return res, dict(zip(res, range(len(res)))), dict(zip(range(len(res)),res))

def get_ckpt_filename(name, epoch):
    return '{}-{}.ckpt'.format(name,epoch)