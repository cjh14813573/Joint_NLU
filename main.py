
import argparse
import torch
import os

parser = argparse.ArgumentParser()
###
parser.add_argument('--bert_path', help='config file', default='/Users/cjh/develop/python/AI/models/bert-base-chinese')
parser.add_argument('--save_path', help='path to save checkpoint', default='./save')
parser.add_argument('--lr', help='learning rate', type=float, default=8e-6)
parser.add_argument('--lr_warmup', type=float, default=200)
parser.add_argument('--bs', help='batch size', type=float, default=30)
parser.add_argument('--n_jobs', help='num of workers to process data', type=int, default=1)
###

args = parser.parse_args()

os.environ['CUDA_VISABLE_DEVICES'] = args.gpu

from transformers import BertConfig, BertTokenizer, AdamW
from NLU_model import NLUModel
import dataset
import utils

train_path = os.path.join(args.save_path, 'train')
log_path = os.path.join(args.save_path, 'log')

def save_func(epoch, device):
    filename = utils.get_ckpt_filename('model',epoch)
    torch.save(trainer.state_dict(), os.path.join(train_path, filename))






