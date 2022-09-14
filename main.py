from focalloss import *
import horovod.torch as hvd
from sklearn.metrics import precision_recall_curve
from trash.load_database import *
import random
import warnings
from torch.utils.data import Dataset
from model import *
import torch.nn as nn
import torch
warnings.filterwarnings('ignore')
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from argparse import ArgumentParser
import pickle
import warnings
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from mini_batch_loader import MyNeighborSampler
from torch_geometric.data import Data, Batch, NeighborSampler
from torch_geometric.utils import to_undirected, add_self_loops
debug=True
import torch.sparse
warnings.filterwarnings('ignore')
if __name__=='__main__':


    if torch.cuda.is_available():
        import horovod.torch as hvd
        hvd.init()
        hvd_rank = hvd.rank()
        hvd_size = hvd.size()
        hvd_local_rank = hvd.local_rank()
        device = torch.device('cuda', hvd_rank)
    is_master = (hvd_rank == 0)


class News_Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json', help='config path')

    parser.add_argument('--flag', type=str, default='default', help='any description')
    parser.add_argument('--n_max_posts', type=int, default=1000, help='maximum node number')
    parser.add_argument('--data_split', type=str,default='random',help='topic-based or random')
    parser.add_argument('--mask_tweet',type=int,default=0,help='whether mask the content of tweet and retweet')
    # model parameters
    parser.add_argument('--model_name', type=str, default='test', help='Model name')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden size')
    parser.add_argument('--feat_type', type=str, default='pt+pm+ut+um', help='feat_type')
    parser.add_argument('--text_type', type=str, default='w2v', help='text_type')
    parser.add_argument('--text_encoder', type=str, default='cnn', help='text_encoder')
    parser.add_argument('--network_type', type=str, default='gnn', help='network type')
    parser.add_argument('--user_gnn_type',type=str,default='sage',help='user gnn type')
    parser.add_argument('--pool_type',type=str,default='mean',help='pool type')
    parser.add_argument('--agg_type',type=str,default='att')
    parser.add_argument('--dropout_rate',type=float,default=0.1,help='dropout rate')
    parser.add_argument('--bidirectional',type=int,default=0,help='bidirectional')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--topic_loss_weight',type=float,default=0.5,help='topic loss weight')
    parser.add_argument('--combine_type',type=str,default='split',help='transformer')
    parser.add_argument('--topic_tranductive', type=int, default=0, help='whether use val&test data')
    parser.add_argument('--joint_sample_tranductive', type=int, default=0, help='whether use val&test data')
    parser.add_argument('--joint_loss_weight',type=float,default=0.0,help='joint loss weight')
    parser.add_argument('--n_cross_layer',type=int,default=1,help='the number of cross layer')
    parser.add_argument('--node_filter',type=int,default=0,help='whether use node filter')
    parser.add_argument('--drop_mode',type=str,default='0+0+0')#truncate 0.5, edge drop 0.2, tweet/retweet mask
    parser.add_argument('--dump_feats',type=int,default=0)
    #parser.add_argument('--joint',type=int,default=0,help='whether train jointly')

    parser.add_argument('--batch_size',type=int,default=4,help='batch_size')
    parser.add_argument('--show',type=int,default=0,help='is show')

    # training parameters
    parser.add_argument('--rand_seed', type=int, default=1025, help='rando')
    args = parser.parse_args()

    model_path = './checkpoints/checkpoint_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.pt' % \
                 (args.data_split,
                  args.drop_mode,
                  args.model_name,
                  args.text_type,
                  args.text_encoder,
                  args.node_filter,
                  args.network_type,
                  args.pool_type,
                  args.agg_type,
                  args.user_gnn_type,
                  args.combine_type,
                  args.n_cross_layer,
                  args.bidirectional,
                  args.topic_loss_weight,
                  args.alpha,
                  args.hidden_size,
                  args.topic_tranductive,
                  args.joint_sample_tranductive,
                  args.joint_loss_weight
                  )
    log_path = './logs/xxxx_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_pppp.pt' % \
                 (args.data_split,
                  args.drop_mode,
                  args.model_name,
                  args.text_type,
                  args.text_encoder,
                  args.node_filter,
                  args.network_type,
                  args.pool_type,
                  args.agg_type,
                  args.user_gnn_type,
                  args.combine_type,
                  args.n_cross_layer,
                  args.bidirectional,
                  args.topic_loss_weight,
                  args.alpha,
                  args.hidden_size,
                  args.topic_tranductive,
                  args.joint_sample_tranductive,
                  args.joint_loss_weight
                  )

    setattr(args, 'model_path', model_path)
    setattr(args, 'log_path', log_path)
    return args


#加载feat_matrix后，与0向量进行拼接





def truncate_data_v2(inst,max_n_posts, max_time=0):#with user graph
    news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, p_user_relations, p2p_edges_all,u2p_edges_all,p2p_edge_dist,data_name, data_name_v2, data_name_combined, data_name_combined_v2 = inst

    n_posts=len(p_post_ids)
    if args.truncate_rate>0:
        truncate_n = random.randint(int(n_posts*(1-args.truncate_rate)),n_posts)
        max_n_posts=min(truncate_n,max_n_posts)

    times = raw_post_metas[p_post_ids][:, -2]
    if max_time>0:
        time_n_posts = (times<max_time).sum()
        max_n_posts = min(time_n_posts,max_n_posts)


    if len(p_post_ids)>max_n_posts:
        t_post_ids = p_post_ids[:max_n_posts]
        t_post_types=post_types[:max_n_posts]
        t_aligned_user_ids = p_aligned_user_ids[:max_n_posts]
        post_ids_set = set(t_post_ids)
        assert len(post_ids_set)<=max_n_posts
        t_retweet_relations = [rel for rel in p_retweet_relations if (rel[0] in post_ids_set and rel[1] in post_ids_set)]
        t_reply_relations = [rel for rel in p_reply_relations if (rel[0] in post_ids_set and rel[1] in post_ids_set)]
        #print('p_write',len(p_write_relations))
        t_write_relations = [rel for rel in p_write_relations if rel[1] in post_ids_set]
        user_ids_set = set([rel[0] for rel in t_write_relations])
        t_user_ids = [uid for uid in p_user_ids if uid in user_ids_set]
        t_user_relations = [rel for rel in p_user_relations if rel[0] in t_user_ids and rel[1] in t_user_ids]


        p2p_edges_all = [rel for rel in p2p_edges_all if (rel[0] in post_ids_set and rel[1] in post_ids_set)]
        p2p_edge_dist = [p2p_edge_dist[i] for i,rel in enumerate(p2p_edges_all) if (rel[0] in post_ids_set and rel[1] in post_ids_set)]
        u2p_edges_all = [rel for rel in u2p_edges_all if (rel[0] in user_ids_set and rel[1] in post_ids_set)]

        t_root_pids=[pid for pid in p_root_pids if pid in post_ids_set]

        return news_id, labels, t_post_ids, t_user_ids, t_aligned_user_ids, t_root_pids, t_post_types, t_retweet_relations, t_reply_relations, t_write_relations,t_user_relations, p2p_edges_all,u2p_edges_all,p2p_edge_dist,data_name, data_name_v2, data_name_combined, data_name_combined_v2

    else:
        return news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations,p_user_relations,p2p_edges_all,u2p_edges_all, p2p_edge_dist,data_name, data_name_v2, data_name_combined, data_name_combined_v2


def sample_neighbors(nids,adj,n_nebs=10):
    #print(nids)
    n=adj.shape[0]
    news_id_list=[]
    news_cluster=[]
    target_ids=[]
    offset=0
    for i,nid in enumerate(nids):
        nnz=(adj[nid]>0).sum()
        if nnz>0:
            p = np.abs(adj[nid])/np.abs(adj[nid]).sum()#
            try:
                nebs=np.random.choice(a=n,size=min(nnz,n_nebs),replace=False,p=p).tolist()
            except:
                print('failed',nid)
                assert False
        else:
            nebs=[]
        news_id_list.extend([nid]+nebs)
        news_cluster.extend([i]*(len(nebs)+1))
        target_ids.append(offset)
        offset=len(news_id_list)

    return news_id_list,news_cluster,target_ids



topic_label_dict={'P&E':0,'Health':1,'Covid':2,'Syria':3}


def local_multi_graph_collate_fn_with_global_user(inst_list, post_types,post_metas, user_metas, posts_vecs, users_vecs,nn_adj, max_n_posts=5000,mode='train',max_time=0):


    if 'joint' in args.model_name:
        nids = [news2id_map[x[0]] for x in inst_list]

        news_id_list, news_cluster,target_ids = sample_neighbors(nids, nn_adj.numpy(), n_nebs=10)

        inst_list = [p_data_list[nid] for nid in news_id_list]


    debug=False
    post_Data_list=[]
    user_Data_list=[]
    node_offset=0

    global train_nid_set



    write_edges_list=[]
    response_edges_list=[]
    p_offset=0
    u_offset=0
    for index,inst in enumerate(inst_list):


        t_inst = truncate_data_v2(inst,max_n_posts,max_time)

        #t_inst=inst
        news_id, labels, p_post_ids, p_user_ids,  p_aligned_user_ids, p_root_pids,post_types, p_retweet_relations, p_reply_relations, p_write_relations,p_user_relations,p2p_edges_all,u2p_edges_all, p2p_edge_dist,data_name, data_name_v2, data_name_combined, data_name_combined_v2=t_inst






        if mode=='train' and (news_id not in train_nid_set):
            labels=-1
        assert len(p_user_ids)==len(set(p_user_ids))
        assert len(p_post_ids)==len(post_types)
        assert len(p_post_ids)==len(p_aligned_user_ids)
        
        if debug:

            print(len(p_post_ids),len(p_user_ids),len(p_user_ids))

        post_meta_feats = post_metas[p_post_ids]  # 
        user_meta_feats = user_metas[p_user_ids]  # 



        post_text_feats = posts_vecs[p_post_ids]  # 
        user_text_feats = users_vecs[p_user_ids]  # 


        if args.mask_tweet_rate>0:
            flag = random.randint(0,100)<100*args.mask_tweet_rate
            if flag:
                post_text_feats[post_types != 2] = 0  #


        if debug:
            print(post_text_feats.shape,post_meta_feats.shape,user_text_feats.shape,user_meta_feats.shape)




        local_pid_map={}
        local_uid_map={}
        for i,pid in enumerate(p_post_ids):
            local_pid_map[pid]=i
        for i,uid in enumerate(p_user_ids):
            local_uid_map[uid]=i#

        n_posts = len(local_pid_map)


        puids = [local_uid_map[i] for i in p_aligned_user_ids]


        multi_flag=True
        if multi_flag:
            post_edge_index = p2p_edges_all
            p_write_relations = u2p_edges_all
        else:
            post_edge_index = p_retweet_relations+p_reply_relations



        post_edge_index = [(local_pid_map[r[0]],local_pid_map[r[1]]) for r in post_edge_index]
        write_edge_index = [(local_uid_map[r[0]], local_pid_map[r[1]]) for r in p_write_relations]



        if multi_flag:
            offset = u_offset+p_offset

            write_edge_index = [(r[0]+n_posts+offset,r[1]+offset) for r in write_edge_index]
            write_edges_list.extend(write_edge_index)

        else:
            response_dict=dict()
            for k,v in post_edge_index:
                response_dict[k]=v
            response_edge_index = list(set([(x[0],response_dict[x[1]]) for x in write_edge_index if x[1] in response_dict ]))

            write_edge_index = [(r[0]+u_offset,r[1]+p_offset) for r in write_edge_index]
            response_edge_index = [(r[0] + u_offset, r[1] + p_offset) for r in response_edge_index]
            write_edges_list.extend(write_edge_index)
            response_edges_list.extend(response_edge_index)



        #
        post_edge_index=torch.tensor(post_edge_index,dtype=torch.long).T
        user_edge_index=[(local_uid_map[r[0]],local_uid_map[r[1]]) for r in p_user_relations]
        user_edge_index=torch.tensor(user_edge_index,dtype=torch.long).T

        if debug:
            print('n_post,n_pedges,n_user,n_udges',len(post_ids),post_edge_index.shape[1],len(user_ids),user_edge_index.shape[1])

        if args.network_type!='rgcn':

            post_edge_index=add_self_loops(post_edge_index,num_nodes=post_text_feats.shape[0])[0]#,edge_attr=p2p_edge_dist,fill_value=0)#[0]


            p2p_edge_dist=torch.cat([torch.tensor(p2p_edge_dist),torch.ones(post_edge_index.shape[1]-len(p2p_edge_dist))]).long()



            user_edge_index=add_self_loops(user_edge_index,num_nodes=user_text_feats.shape[0])[0]


        assert p2p_edge_dist.shape[0]==post_edge_index.shape[1]


        topic_y=topic_label_dict[data_name_combined_v2.split('-')[0]]

        post = Data(t=torch.tensor(post_text_feats,dtype=torch.float),
                    m=torch.tensor(post_meta_feats,dtype=torch.float),
                    post_types = torch.tensor(post_types,dtype=torch.long),
                    edge_index=post_edge_index,
                    y=torch.tensor(labels,dtype=torch.long),
                    ty=torch.tensor(topic_y),
                    puids=torch.tensor(puids),
                    num_nodes=post_text_feats.shape[0],
                    edge_dist=p2p_edge_dist)

        user = Data(t=torch.tensor(user_text_feats,dtype=torch.float),
                    m=torch.tensor(user_meta_feats,dtype=torch.float),
                    edge_index=user_edge_index,
                    y=torch.tensor(labels,dtype=torch.long),
                    user_ids=torch.tensor(p_user_ids),
                    num_nodes=user_text_feats.shape[0])
        post_Data_list.append(post)
        user_Data_list.append(user)


        p_offset+=post_text_feats.shape[0]
        u_offset+=user_text_feats.shape[0]


    post_batch=Batch.from_data_list(post_Data_list)
    user_batch=Batch.from_data_list(user_Data_list)


    write_edges_list = torch.tensor(write_edges_list).T
    response_edges_list = torch.tensor(response_edges_list).T

    if 'joint' in args.model_name:
        post_batch.tty=post_batch.ty[target_ids]
        return post_batch,user_batch, torch.tensor(news_cluster),write_edges_list,response_edges_list, post_batch.y[target_ids]
    else:
        return post_batch,user_batch, write_edges_list,response_edges_list, post_batch.y





def get_loss_func(args):

    criterion = nn.BCEWithLogitsLoss()

    return criterion


def mprint(*input):
    if hvd_rank==0:
        print(*input)




def load_post_user_feats_and_embeddings(text_type,mask_tweets=False):
    print('load feats and embeddings')
    post_emb=None
    user_emb=None
    post_metas = load_npy('./datasets/post_metas.npy')
    user_metas = load_npy('./datasets/user_metas.npy')
    post_ids,user_ids,post_types=load_pkl('./datasets/ids.pkl')


    if text_type=='tfidf':
        posts_vecs=load_pkl('./datasets/tweet_tfidf_feat.pkl')
        users_vecs=load_pkl('./datasets/user_tfidf_feat.pkl')
    elif text_type=='w2v':
        posts_vecs=load_npy('./datasets/tweet_w2v_feat.npy')
        users_vecs=load_npy('./datasets/user_w2v_feat.npy')
    elif text_type=='bert':
        posts_vecs=load_npy('./datasets/tweet_bert_feat.npy')
        users_vecs=load_npy('./datasets/user_bert_feat.npy')
    elif text_type=='wids-split':
        posts_vecs=load_npy('./datasets/split_post_token_ids.npy')
        users_vecs=load_npy('./datasets/split_user_token_ids.npy')
        post_emb=load_npy('./datasets/split_post_embedding.npy')
        user_emb=load_npy('./datasets/split_user_embedding.npy')
    elif text_type=='wids-share':
        posts_vecs=load_npy('./datasets/share_post_token_ids.npy')
        users_vecs=load_npy('./datasets/share_user_token_ids.npy')
        post_emb=load_npy('./datasets/share_all_embedding.npy')
        user_emb=None
    else:
        raise ValueError('feature type error!')

    if mask_tweets:
        for i,pt in enumerate(post_types):
            if pt!=3:
                vec = np.zeros(100,dtype=np.long)
                vec[0]=pt
                posts_vecs[i]=vec

    return post_ids,user_ids,post_types,post_metas,user_metas,posts_vecs,users_vecs,post_emb,user_emb

def print_metrics(labels,preds,probs,best_threshold=None):

    labels=np.array(labels)
    preds=np.array(preds)
    probs=np.array(probs)

    f1=f1_score(labels, preds)

    if best_threshold is None:
        precisions,recalls,thresholds=precision_recall_curve(labels,probs)
        f1s = 2*precisions*recalls/(precisions+recalls)
        f1s=np.nan_to_num(f1s)
        max_id = f1s.argmax()
        best_threshold = thresholds[max_id]
        best_f1=f1s[max_id]
        assert best_f1==f1s.max()
    else:
        preds[probs>=best_threshold]=1
        preds[probs<best_threshold]=0
        best_f1=f1_score(labels, preds)



    auc = roc_auc_score(labels, probs)

    print('f1',f1)

    print('best f1',best_f1)
    print('auc', auc)
    return auc,f1,best_f1,best_threshold



train_nid_set=set()


if __name__=='__main__':

    sigmoid = nn.Sigmoid()
    num_epochs = 30
    over_write = False
    min_n_users = 1
    min_n_posts = 10



    args = parse_args()
    if args.text_type in ['tfidf', 'w2v', 'bert']:
        assert args.text_encoder == 'none'
    show = int(args.show)

    truncate_rate, edge_drop_rate, mask_tweet_rate = args.drop_mode.split('+')
    truncate_rate = float(truncate_rate)
    edge_drop_rate = float(edge_drop_rate)
    mask_tweet_rate = float(mask_tweet_rate)
    args.truncate_rate=truncate_rate
    args.edge_drop_rate=edge_drop_rate
    args.mask_tweet_rate=mask_tweet_rate


    criterion = get_loss_func(args)
    print('loading processed dataset')
    f = open('./datasets/processed_context_data_v4.pkl', 'rb')#with edge dist

    p_data_list = pickle.load(f)

    post_ids, user_ids, post_types,post_metas, user_metas, posts_vecs, users_vecs, post_emb, user_emb=\
        load_post_user_feats_and_embeddings(args.text_type)

    raw_post_metas = post_metas

    #normalization
    post_metas = (post_metas-post_metas.mean(0))/post_metas.std(0)


    #user_metas = load_npy('./datasets/processed_user_feats.npy')
    user_metas = (user_metas-user_metas.mean(0))/user_metas.std(0)

    post_type_onehot = np.zeros([len(post_types),3])
    for i,pt in enumerate(post_types):
        post_type_onehot[i][pt-1]=1
    post_metas = np.concatenate([post_metas,post_type_onehot],axis=1)

    if args.user_gnn_type != 'none':
        mprint('loading processed_edges.pkl')
        p_edges = load_npy('./datasets/processed_edges.npy')

        neb_sampler = MyNeighborSampler(torch.tensor(p_edges).T, node_idx=None,
                                       sizes=[20], batch_size=64,
                                       shuffle=True, num_workers=12)




    news_id_list=[x[0] for x in p_data_list]
    news2id_map={}
    for nid,news_id in enumerate(news_id_list):
        news2id_map[news_id]=nid
    nn_adj = torch.tensor(load_npy('./datasets/engage_nn_adj.npy')).float()



    #split
    data_set_list = data_split(p_data_list,split_mode=args.data_split)
    if args.data_split=='random':
        assert 'story_reviews_01116'==data_set_list[0][0][0][0]
    else:
        assert 'gossipcop-916069' == data_set_list[0][0][0][0]





    for i in range(5):

        for data_set_id,(train_list,val_list,test_list) in enumerate(data_set_list):






            try:
                early_stopping = EarlyStopping(patience=4, verbose=True, path=args.model_path, trace_func=print,
                                               rank=hvd_rank)

                #if args.mask_tweet:
                #    masked_train_list = []
                #    for inst in train_list:
                #        news_id=inst[0]+'masktweets'
                #        masked_train_list.append(tuple([news_id])+inst[1:])
                #    train_list+=masked_train_list


                train_set = News_Dataset(train_list)
                val_set = News_Dataset(val_list)
                test_set = News_Dataset(test_list)
                all_set = News_Dataset(train_list+val_list+test_list)

                train_nid_set = set([x[0] for x in train_list])
                val_nid_set = set([x[0] for x in val_list])
                test_nid_set = set([x[0] for x in test_list])

                def mask_edges(train_nid_set,val_nid_set,test_nid_set,nn_adj):
                    train_nid_list = [news2id_map[nid] for nid in list(train_nid_set)]
                    val_nid_list = [news2id_map[nid] for nid in list(val_nid_set)]
                    test_nid_list = [news2id_map[nid] for nid in list(test_nid_set)]
                    train_mask_list = val_nid_list+test_nid_list
                    val_mask_list = test_nid_list

                    train_nn_adj=nn_adj.clone()
                    val_nn_adj=nn_adj.clone()
                    test_nn_adj=nn_adj.clone()
                    train_nn_adj[:,train_mask_list]=0
                    val_nn_adj[:,val_mask_list]=0
                    return train_nn_adj,val_nn_adj,test_nn_adj

                if args.joint_sample_tranductive:
                    train_nn_adj=val_nn_adj=test_nn_adj=nn_adj
                else:
                    train_nn_adj,val_nn_adj,test_nn_adj = mask_edges(train_nid_set,val_nid_set,test_nid_set,nn_adj)




                #train_loader = DataLoader(train_set,batch_size=4*hvd_size,shuffle=True,collate_fn=lambda x:tf_collate_fn(x,t_token_ids_map, u_token_ids_map, t_meta_map,u_meta_map))
                #test_loader = DataLoader(test_set, batch_size=4*hvd_size, shuffle=False,collate_fn=lambda x:tf_collate_fn(x,t_token_ids_map, u_token_ids_map, t_meta_map,u_meta_map))

                train_loader = DataLoader(train_set,batch_size=args.batch_size*hvd_size,shuffle=True,collate_fn=lambda x:local_multi_graph_collate_fn_with_global_user(x, post_types,post_metas, user_metas, posts_vecs, users_vecs,train_nn_adj, max_n_posts=args.n_max_posts,mode='train'),num_workers=4)
                val_loader = DataLoader(val_set, batch_size=args.batch_size*hvd_size, shuffle=False,collate_fn=lambda x:local_multi_graph_collate_fn_with_global_user(x, post_types,post_metas, user_metas, posts_vecs, users_vecs, val_nn_adj,max_n_posts=args.n_max_posts,mode='val'),num_workers=4)
                test_loader = DataLoader(test_set, batch_size=args.batch_size*hvd_size, shuffle=False,collate_fn=lambda x:local_multi_graph_collate_fn_with_global_user(x, post_types,post_metas, user_metas, posts_vecs, users_vecs,test_nn_adj, max_n_posts=args.n_max_posts,mode='test'),num_workers=4)


                labels = []
                preds = []
                probs = []
                feat_list = []
                with torch.no_grad():
                    for batch in tqdm(test_loader):
                        labels_batch = batch[-1] if isinstance(batch, list) or isinstance(batch, tuple) else batch.y
                        labels.extend(labels_batch.tolist())
                dump_npy(np.array(labels), 'test_labels_%s.npy' % data_set_id)

                #continue



                if args.topic_tranductive:
                    train_loader = DataLoader(all_set,batch_size=args.batch_size*hvd_size,shuffle=True,collate_fn=lambda x:local_multi_graph_collate_fn_with_global_user(x, post_types,post_metas, user_metas, posts_vecs, users_vecs,train_nn_adj, max_n_posts=args.n_max_posts),num_workers=4)


                args.post_meta_size = post_metas.shape[1]
                args.user_meta_size = user_metas.shape[1]
                #model = Transformer_Model_with_encoder(args).to(device)

                model = Multi_View_Graph_model_with_cross_network(args,tweet_emb=post_emb,user_emb=user_emb).to(device)


                if args.network_type=='transformer':
                    optimizer = AdamW(model.parameters(), lr=5e-5)
                else:
                    optimizer = Adam(model.parameters())


                optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(optimizer, root_rank=0)


                barrier(hvd)




                val_results=[]
                for epoch in range(num_epochs):
                    mprint('Epoch %s:================' % epoch)
                    model.train()
                    if show:
                        train_loader = tqdm(train_loader) if hvd_size == 1 else train_loader
                    labels = []
                    preds = []
                    probs = []
                    all_loss=[]
                    for batch in train_loader:
                        batch = [item.to(device) if not isinstance(item, list) else item for item in batch] if isinstance(
                            batch, tuple) or isinstance(batch,list) else batch.to(device)

                        labels_batch = batch[-1] if isinstance(batch,list) else batch.y

                        x_batch = batch[:-1] if isinstance(batch,list) else batch
                        outputs_batch,topic_out_batch,joint_outputs_batch = model(x_batch)
                        #print(outputs_batch)
                        #a=input(' ')
                        probs_batch = sigmoid(outputs_batch)  # nx1 BCE
                        preds_batch = (probs_batch>0.5).long()

                        if (labels_batch>=0).sum()>0:
                            veracity_loss = criterion(outputs_batch[labels_batch>=0], labels_batch[labels_batch>=0].float())
                        else:
                            veracity_loss = 0

                        if args.topic_loss_weight>0:
                            if topic_out_batch.shape[0]==batch[0].ty.shape[0]:
                                topic_loss = args.topic_loss_weight*F.cross_entropy(topic_out_batch,batch[0].ty)
                            else:
                                topic_loss = args.topic_loss_weight * F.cross_entropy(topic_out_batch, batch[0].tty)
                        else:
                            topic_loss=0

                        if 'joint' in args.model_name and args.joint_loss_weight>0:
                            if 'target' in args.model_name:
                                joint_loss = args.joint_loss_weight * criterion(joint_outputs_batch[labels_batch>=0],labels_batch[labels_batch>=0].float())
                            else:#all
                                joint_loss = args.joint_loss_weight * criterion(joint_outputs_batch[batch[0].y>=0],batch[0].y[batch[0].y>=0].float())
                        else:
                            joint_loss=0



                        loss = veracity_loss + topic_loss + joint_loss

                        loss.backward()
                        all_loss.append(loss.item())
                        if isinstance(train_loader,tqdm):
                            train_loader.set_description('Loss %.5f| A:%.5f V:%.5f J:%.5f T:%.5f'%(np.mean(all_loss),loss,float(veracity_loss),float(joint_loss),float(topic_loss)))
                        optimizer.step()
                        #scheduler.step()
                        optimizer.zero_grad()
                        labels.extend(labels_batch[labels_batch>=0].tolist())
                        preds.extend(preds_batch[labels_batch>=0].tolist())
                        probs.extend(probs_batch[labels_batch>=0].tolist())
                        ##progress_bar.update(1)
                    mprint('Training metrics--------')
                    all_labels, all_preds, all_probs = gather_all_outputs(labels, preds, probs, hvd)
                    print_metrics(all_labels, all_preds, all_probs)



                    if show:
                        val_loader = tqdm(val_loader) if hvd_size==1 else val_loader
                    labels = []
                    preds = []
                    probs = []
                    model.eval()
                    with torch.no_grad():
                        for batch in val_loader:

                            batch = [item.to(device) if not isinstance(item,list) else item for item in batch] if isinstance(batch,tuple) else batch.to(device)
                            labels_batch = batch[-1] if isinstance(batch,list) else batch.y
                            x_batch = batch[:-1] if isinstance(batch,list) else batch
                            outputs_batch,_,_ = model(x_batch)#.reshape(-1)


                            probs_batch = sigmoid(outputs_batch)  # nx1 BCE
                            preds_batch = (probs_batch > 0.5).long()
                            labels.extend(labels_batch.tolist())
                            preds.extend(preds_batch.tolist())
                            probs.extend(probs_batch.tolist())
                            # metric.add_batch(predictions=predictions, references=batch["labels"])

                    all_labels,all_preds,all_probs=gather_all_outputs(labels,preds,probs, hvd)
                    mprint('Validation metrics--------')
                    auc,f1,best_f1,threshold  = print_metrics(all_labels, all_preds, all_probs)

                    if len(val_results)==0 or auc > max(val_results):
                        best_threshold=threshold

                    val_results.append(auc)
                    early_stopping(auc, model)
                    if early_stopping.early_stop:
                        print('Early stopping...........................')
                        break





                    #=====test
                    if args.data_split=='xxx':
                        if show:
                            test_loader = tqdm(test_loader) if hvd_size == 1 else test_loader
                        labels = []
                        preds = []
                        probs = []
                        model.eval()
                        with torch.no_grad():
                            for batch in test_loader:
                                batch = [item.to(device) if not isinstance(item, list) else item for item in
                                         batch] if isinstance(batch, tuple) else batch.to(device)

                                labels_batch = batch[-1] if isinstance(batch, list) else batch.y
                                x_batch = batch[:-1] if isinstance(batch, list) else batch
                                outputs_batch,_,_ = model(x_batch).reshape(-1)

                                probs_batch = sigmoid(outputs_batch)  # nx1 BCE
                                preds_batch = (probs_batch > 0.5).long()
                                labels.extend(labels_batch.tolist())
                                preds.extend(preds_batch.tolist())
                                probs.extend(probs_batch.tolist())
                            # metric.add_batch(predictions=predictions, references=batch["labels"])

                        all_labels, all_preds, all_probs = gather_all_outputs(labels, preds, probs, hvd)
                        mprint('test metrics--------')
                        test_auc,test_f1 = print_metrics(all_labels, all_preds, all_probs)





                #load best model
                print('++++++++++++++load best model++++++++++++++++++++++++++++++++')
                model.load_state_dict(torch.load(args.model_path))
                #n_max_posts_list = [10,20,30,50,100,args.n_max_posts]
                max_time_list = [3600*(x+1) for x in range(10)]+[3600*100000]
                auc_f1_list=[]
                for max_time in max_time_list:
                    #print(n_max_post)
                    print(max_time)
                    test_loader = DataLoader(test_set, batch_size=args.batch_size*hvd_size, shuffle=False,collate_fn=lambda x:local_multi_graph_collate_fn_with_global_user(x, post_types,post_metas, user_metas, posts_vecs, users_vecs,test_nn_adj, max_n_posts=2000,max_time=max_time,mode='test'),num_workers=4)



                    if show:
                        test_loader = tqdm(test_loader) if hvd_size == 1 else test_loader
                    labels = []
                    preds = []
                    probs = []
                    model.eval()
                    with torch.no_grad():
                        for batch in test_loader:
                            batch = [item.to(device) if not isinstance(item, list) else item for item in batch] if isinstance(
                                batch, tuple) else batch.to(device)

                            labels_batch = batch[-1] if isinstance(batch, list) else batch.y
                            x_batch = batch[:-1] if isinstance(batch, list) else batch
                            outputs_batch,_,_ = model(x_batch)#.reshape(-1)

                            probs_batch = sigmoid(outputs_batch)  # nx1 BCE
                            preds_batch = (probs_batch > 0.5).long()
                            labels.extend(labels_batch.tolist())
                            preds.extend(preds_batch.tolist())
                            probs.extend(probs_batch.tolist())
                            # metric.add_batch(predictions=predictions, references=batch["labels"])

                    all_labels, all_preds, all_probs = gather_all_outputs(labels, preds, probs, hvd)



                    mprint('test metrics--------')
                    test_auc,test_f1,best_test_f1,_ = print_metrics(all_labels, all_preds, all_probs,best_threshold)
                    if os.path.exists(args.log_path.replace('xxxx','fail').replace('pppp',str(data_set_id))):
                        os.system('rm '+args.log_path.replace('xxxx','fail').replace('pppp',str(data_set_id)))

                    auc_f1_list.append([test_auc,test_f1])

                    if  max_time==3600*100000:#n_max_post==args.n_max_posts:
                        f=open(args.log_path.replace('xxxx','%.4f_%.4f_%.4f'%(test_auc,best_test_f1,test_f1)).replace('pppp',str(data_set_id)),'w')
                        f.write('\n'.join([str(x) for x in val_results]))
                        f.close()



                fname= args.log_path.replace('xxxx', 'early%.4f_%.4f_%.4f' % (test_auc, best_test_f1, test_f1)).replace('pppp',str(data_set_id))
                np.savetxt(fname,np.array(auc_f1_list),fmt='%.4f')



                if True:
                    print('++++++++++++++load best model++++++++++++++++++++++++++++++++')
                    model.load_state_dict(torch.load(args.model_path))
                    # n_max_posts_list = [10,20,30,50,100,args.n_max_posts]

                    # print(n_max_post)
                    #print(max_time)
                    test_loader = DataLoader(test_set, batch_size=args.batch_size * hvd_size, shuffle=False,
                                             collate_fn=lambda x: local_multi_graph_collate_fn_with_global_user(x,
                                                                                                                post_types,
                                                                                                                post_metas,
                                                                                                                user_metas,
                                                                                                                posts_vecs,
                                                                                                                users_vecs,
                                                                                                                test_nn_adj,
                                                                                                                max_n_posts=2000,
                                                                                                                max_time=0,
                                                                                                                mode='test'),
                                             num_workers=4)

                    if show:
                        test_loader = tqdm(test_loader) if hvd_size == 1 else test_loader
                    labels = []
                    preds = []
                    probs = []
                    feat_list=[]
                    model.eval()
                    with torch.no_grad():
                        for batch in test_loader:
                            batch = [item.to(device) if not isinstance(item, list) else item for item in
                                     batch] if isinstance(
                                batch, tuple) else batch.to(device)

                            labels_batch = batch[-1] if isinstance(batch, list) or isinstance(batch, tuple) else batch.y
                            x_batch = batch[:-1] if isinstance(batch, list) else batch
                            outputs_batch, _, _,feats = model(x_batch,extract_feat=True)  # .reshape(-1)
                            feat_list.append(feats)
                            probs_batch = sigmoid(outputs_batch)  # nx1 BCE
                            preds_batch = (probs_batch > 0.5).long()
                            labels.extend(labels_batch.tolist())
                            preds.extend(preds_batch.tolist())
                            probs.extend(probs_batch.tolist())

                    all_feats = torch.cat(feat_list,dim=0).cpu().numpy()
                    all_labels, all_preds, all_probs = gather_all_outputs(labels, preds, probs, hvd)

                    mprint('test metrics--------')


                    test_auc, test_f1, best_test_f1, _ = print_metrics(all_labels, all_preds, all_probs, best_threshold)
                    fname = args.log_path.replace('xxxx',
                                                  'feats%.4f_%.4f_%.4f' % (test_auc, best_test_f1, test_f1)).replace(
                        'pppp', str(data_set_id)).replace('.pt','.npy')

                    dump_npy(all_feats, fname)








            except Exception as e:

                f=open(args.log_path.replace('xxxx','fail').replace('pppp',str(data_set_id)),'w')
                f.close()
                traceback.print_exc()
                assert 1 == 0




