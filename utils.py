# Tweepy
# Copyright 2009-2021 Joshua Roesslein
# See LICENSE for details.

import h5py
import time
import argparse
import itertools
from tqdm import tqdm
import json
import numpy as np
import pickle
import pandas as pd
import h5py
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import torch
import traceback
import random
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,precision_score,recall_score
import torch.nn as nn
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split


from tweepy.mixins import DataMapping, HashableID

if torch.cuda.is_available():
    import horovod.torch as hvd
    hvd.init()
    hvd_rank = hvd.rank()
    hvd_size = hvd.size()
    hvd_local_rank = hvd.local_rank()
    is_master = (hvd_rank == 0)
    device = torch.device('cuda', hvd_rank)
else:
    device = torch.device('cpu')


sigmoid=nn.Sigmoid()


class Time_Count(HashableID, DataMapping):

    __slots__ = ("data","start", "end", "tweet_count")

    def __init__(self, data):
        self.data = data
        self.start = data["start"]
        self.end = data["end"]
        self.tweet_count = data['tweet_count']

    def __repr__(self):
        return f"<Time_Count start={self.start} end={self.end} tweet_count={self.tweet_count}"

def delete_duplicated(data_list):
    id_set = set()
    filtered_data_list = []
    for data in data_list:
        if data['id'] not in id_set:
            filtered_data_list.append(data)
            id_set.add(data['id'])
    return filtered_data_list



tfidf_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
              "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
              'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
               'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
              'the', 'and',  'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
              'about',  'between', 'into', 'through', 'during', 'to', 'from', 'in', 'out', 'on', 'off',
              'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each',
              'other', 'some', 'such', 'own', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
               'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', ]#'what', 'which', 'who', 'whom','but','against',"don't","aren't", "couldn't","didn't","doesn't","hadn't","haven't", 'isn', "isn't",  'when', 'where', 'why', 'how',


def get_user_ids(users_str):
    users=json.loads(users_str)
    user_ids = [x['id'] for x in users]
    return user_ids

def dump_pkl(obj,path):
    f=open(path,'wb')
    pickle.dump(obj,f)
    f.close()

def load_pkl(path):
    f=open(path,'rb')
    obj=pickle.load(f)
    f.close()
    return obj

#def load_json(path):
#    f=open(path,'r')

def dump_npy(obj,path):
    assert path[-4:]=='.npy'
    np.save(path,obj)
def load_npy(path):
    return np.load(path,allow_pickle=True)


def dump_h5(obj_list,path):
    f=h5py.File(path,'w')
    for i, obj in enumerate(obj_list):
        f[str(i)]=obj
    f.close()

def load_h5(path):
    f=h5py.File(path,'r')
    keys=list(f.keys())
    keys = [int(x) for x in keys]
    keys.sort()
    data_list=[]
    for k in keys:
        data_list.append(f[str(k)])
    return data_list



nltk_stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
           "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
           'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
           'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
           'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
           'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
           'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
           'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
           'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
           'about', 'against', 'between', 'into', 'through', 'during', 'before',
           'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
           'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
           'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
           'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
           'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
           'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
           'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
           'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
           'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
           "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
           "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, rank=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.rank = rank
        self.cnt=0#训练迭代数，不是epoch
        self.best_cnt=0
        self.score_list=[]
    def __call__(self, val_acc, model):

        score = val_acc
        self.cnt+=1
        self.score_list.append([self.cnt,val_acc])
        if self.best_score is None:
            self.best_score = 0
            self.best_cnt=self.cnt
            self.save_checkpoint(score, model)
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.rank==0:
                self.trace_func('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:

            self.save_checkpoint(score, model)
            self.best_score = score
            self.best_cnt=self.cnt
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        model.eval()
        if self.rank==0:
            if self.verbose:
                self.trace_func('Validation AUC increased ({} --> {}).  Saving model ...'.format(self.best_score,score))
            torch.save(model.state_dict(), self.path)

    def train(train_loader, model,optimizer,criterion,print_metrics, is_show=True):
        model.train()
        if is_show:
            train_loader = tqdm(train_loader) if is_master else train_loader
        labels = []
        preds = []
        probs = []
        all_loss = []
        for batch in train_loader:
            ##################replace#########################
            batch = [item.to(device) for item in batch] if isinstance(batch, list) else batch.to(device)
            labels_batch = batch[-1] if isinstance(batch, list) else batch.y
            x_batch = batch[:-1] if isinstance(batch, list) else batch
            outputs_batch = model(x_batch).reshape(-1)
            ##################replace##########################
            probs_batch = sigmoid(outputs_batch)  # nx1 BCE
            preds_batch = (probs_batch > 0.5).long()
            loss = criterion(outputs_batch, labels_batch.float())
            loss.backward()
            all_loss.append(loss.item())
            if isinstance(train_loader, tqdm):
                train_loader.set_description('Loss %s' % np.mean(all_loss))
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            labels.extend(labels_batch.tolist())
            preds.extend(preds_batch.tolist())
            probs.extend(probs_batch.tolist())
        print('Training metrics--------')
        # all_labels, all_preds, all_probs = gather_all_outputs(labels, preds, probs, hvd)
        train_auc = print_metrics(labels, preds, probs)
        return train_auc

    def test(test_loader, model,print_metrics, is_show=True):
        if is_show:
            test_loader = tqdm(test_loader) if hvd_size == 1 else test_loader
        labels = []
        preds = []
        probs = []
        #all_loss = []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                ##################replace#########################
                batch = [item.to(device) for item in batch] if isinstance(batch, list) else batch.to(device)
                labels_batch = batch[-1] if isinstance(batch, list) else batch.y
                x_batch = batch[:-1] if isinstance(batch, list) else batch
                ##################replace##########################
                outputs_batch = model(x_batch).reshape(-1)

                probs_batch = sigmoid(outputs_batch)  # nx1 BCE
                preds_batch = (probs_batch > 0.5).long()
                labels.extend(labels_batch.tolist())
                preds.extend(preds_batch.tolist())
                probs.extend(probs_batch.tolist())
            # metric.add_batch(predictions=predictions, references=batch["labels"])
        print('test metrics--------')
        test_auc = print_metrics(labels, preds, probs)
        return test_auc


def data_split(p_data_list,split_mode='random'):
    data_set_list=[]
    if split_mode=='topic':
        #def topic_split(p_data_list):
        topic_splits=[[['P&E','Sryia'],['Health','Covid']],
                     [['Health','Covid'],['P&E','Sryia']],
                     [['P&E','Health'],['Sryia','Covid']],
                     [['Sryia','Covid'],['P&E','Health']]]

        #p_data_df = pd.DataFrame(p_data_list, columns='news_id,labels,post_ids,user_ids,aligned_user_ids,post_types,retweet_relations, reply_relations, write_relations, data_name, data_name_v2, data_name_combined, data_name_combined_v2'.split(','))


        for train_topics,test_topics in topic_splits:
            train_list = [data for data in p_data_list if train_topics[0] in data[-1] or train_topics[1] in data[-1]]
            test_list = [data for data in p_data_list if test_topics[0] in data[-1] or test_topics[1] in data[-1]]

            train_labels = [data[1] for data in train_list]
            train_list,val_list = train_test_split(train_list, test_size=0.2, random_state=1024,stratify=train_labels)

            val_labels = [data[1] for data in val_list]
            train_labels = [data[1] for data in train_list]
            test_labels = [data[1] for data in test_list]
            print(train_topics,test_topics)
            print('train fake/real/total :%s/%s/%s'%(sum(train_labels),len(train_labels)-sum(train_labels),len(train_labels)))
            print('val fake/real/total :%s/%s/%s'%(sum(val_labels),len(val_labels)-sum(val_labels),len(val_labels)))
            print('test fake/real/total :%s/%s/%s'%(sum(test_labels),len(test_labels)-sum(test_labels),len(test_labels)))
            data_set_list.append([train_list,val_list,test_list])
    else:
        #random split
        for random_seed in [1024,1025,1026]:
            labels = [data[1] for data in p_data_list]
            train_val_list, test_list = train_test_split(p_data_list, test_size=0.2, random_state=random_seed, stratify=labels)
            train_val_labels = [data[1] for data in train_val_list]
            train_list,val_list = train_test_split(train_val_list, test_size=0.25, random_state=random_seed,stratify=train_val_labels)
            val_labels = [data[1] for data in val_list]
            train_labels = [data[1] for data in train_list]
            test_labels = [data[1] for data in test_list]
            print('train fake/real/total :%s/%s/%s' % (sum(train_labels), len(train_labels) - sum(train_labels), len(train_labels)))
            print('val fake/real/total :%s/%s/%s' % (sum(val_labels), len(val_labels) - sum(val_labels), len(val_labels)))
            print('test fake/real/total :%s/%s/%s' % (sum(test_labels), len(test_labels) - sum(test_labels), len(test_labels)))
            data_set_list.append([train_list,val_list,test_list])

    return data_set_list


def print_metrics(labels,preds,probs):
    #mprint('acc', accuracy_score(labels, preds))
    #mprint('f1', f1_score(labels, preds))
    #mprint('precision',precision_score(labels,preds,pos_label=1))
    #mprint('recall',recall_score(labels,preds,pos_label=1))
    #mprint('confusion',confusion_matrix(labels,preds))
    f1=f1_score(labels, preds)
    pr=precision_score(labels,preds,pos_label=1)
    rc=recall_score(labels, preds, pos_label=1)
    auc = roc_auc_score(labels, probs)
    print('f1',f1)
    print('precision',pr)
    print('recall',rc)
    print('auc', auc)
    return f1