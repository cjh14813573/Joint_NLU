import torch
from torch.utils.data import Dataset


class NLUDataset(Dataset):
    def __init__(self, paths, tokz, cls_vocab, slot_vocab, logger, max_lengths=2048):
        self.logger = logger
        self.data = NLUDataset.make_dataset(paths, tokz, cls_vocab, slot_vocab, logger, max_lengths)

    @staticmethod
    def make_dataset(paths, tokz, cls_vocal, slot_vocab, logger, max_lengths):
        logger.info('reading data from {}'.format(paths))
        dataset = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
                lines = [i.split('\t') for i in lines]
                for label, utt, slots in lines:
                    utt = tokz.convert_tokens_to_ids(list(utt)[:max_lengths])
                    slots = [slot_vocab[i] for i in slots.split()] #slot_vacab: slot to index
                    assert len(utt) == len(slots)
                    dataset.append([int(cls_vocal[label]),
                                    [tokz.cls_token_id] + utt + [tokz.sep_token_id],
                                    tokz.create_token_type_ids_from_sequences(token_ids_0=utt),
                                    [tokz.pad_token_id] + slots + [tokz.pad_token_id]])
                    #[0]: intent_id
                    #[1]: cls_token_id + utt + seq_token_id
                    #[2]: mask(0,0,0,1,1,1) 此处只有0
                    #[4]: pad_token_id + slots_id + pad_token_id
        logger.info('{} data record loaded'.format(len(dataset)))
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        intent, utt, token_type, slot = self.data[idx]
        return {"intent": intent, "utt": utt, "token_type": token_type, "slot": slot}


class PinnedBatch:
    def __init__(self,data):
        self.data = data
    def __getitem__(self, i):
        return self.data[i]
    def pin_memory(self):
        for k in self.data.keys():
            self.data[k] = self.data[k].pin_memory()
        return self

class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        res['intent'] = torch.LongTensor([i['intent'] for i in batch])
        max_len = max([len(i['utt']) for i in batch])
        res['utt'] = torch.LongTensor([i['utt'] + [self.pad_id] * (max_len - len(i['utt'])) for i in batch])
        res['mask'] = torch.LongTensor([[1] * len(i['utt']) + [0] * (max_len - len(i['utt'])) for i in batch])
        res['token_type'] = torch.LongTensor([i['token_type'] + [self.pad_id] * (max_len - len(i['token_type'])) for i in batch])
        res['slot'] = torch.LongTensor([i['slot'] + [self.pad_id] * (max_len - len(i['slot'])) for i in batch])
        return PinnedBatch(res)




