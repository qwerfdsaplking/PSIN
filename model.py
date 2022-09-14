# -*- coding: utf-8 -*-
import torch
import math
from torch.nn import init
import json
from torch_sparse import coalesce
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from torch_scatter import scatter_mean,scatter_add,scatter_max
from torch_geometric.utils import to_undirected, add_self_loops
#from torch_geometric.nn import *
from torch_geometric.nn import SAGEConv,GATConv,GCNConv,GlobalAttention,GATv2Conv,RGCNConv,ChebConv
from torch.nn.utils.rnn import PackedSequence
import torch.sparse
from torch_sparse import spmm
from rgat_conv import RGATConv
from tree_gat_conv import TreeGATConv
from gradient_reversal import revgrad,GradientReversal

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')



class lstm_encoder(torch.nn.Module):
    def __init__(self,emb,input_size,hidden_size,n_layers, ffn=False):
        super(lstm_encoder,self).__init__()
        self.lstm = nn.GRU(input_size,hidden_size,n_layers,batch_first=True)
        self.emb_layer=nn.Embedding(100000,100,padding_idx=0,_weight=emb)
        self.dropout=nn.Dropout(0.1)
        self.ffn=ffn
        if ffn:
            self.fc1=nn.Linear(hidden_size,hidden_size)
    def forward(self,x):
        #n x len
        x=x.long()
        x_len = x.bool().long().sum(1).detach().cpu()
        x_len_zero = (x_len==0).to(x.device)#handle void texts
        x_len[x_len_zero]=1#handle void texts to fit in pack_padded

        x_len_sort,idx = x_len.sort(0,descending=True)

        _,un_idx = torch.sort(idx.to(x.device),dim=0)

        x_emb = self.emb_layer(x)
        x_packed = nn.utils.rnn.pack_padded_sequence(input=x_emb,lengths=x_len_sort,batch_first=True)

        x_out,_=self.lstm(x_packed)# n x len xinput_size
        x_out,_ = nn.utils.rnn.pad_packed_sequence(x_out,batch_first=True)#total_length=len
        x_out = torch.index_select(x_out,0,un_idx)

        x_len[x_len == 0] = 1#防止len==0导致nan
        x_pool = x_out.mean(1).squeeze()/x_len.to(x.device).unsqueeze(1)
        x_pool[x_len_zero]=0#recover void texts as 0 vector
        x_out=self.dropout(x_pool)
        if self.ffn:
            x_out = F.relu(self.fc1(x_out))

        return x_out

class cnn_encoder(torch.nn.Module):
    def __init__(self, emb,emb_size, kernel_num, hidden_size=100, kernel_sizes=(2,3,4),ffn=False):
        super(cnn_encoder,self).__init__()
        self.emb_layer = nn.Embedding(100000, 100, padding_idx=0, _weight=emb)
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=kernel_num,kernel_size=(2,emb_size))
        self.conv3 = nn.Conv2d(in_channels=1,out_channels=kernel_num,kernel_size=(3,emb_size))
        self.conv4 = nn.Conv2d(in_channels=1,out_channels=kernel_num,kernel_size=(4,emb_size))
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels=1,out_channels=kernel_num,kernel_size=(ks,emb_size)) for ks in kernel_sizes])
        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, hidden_size)
    def forward(self,x, post_types=None):
        x=x.long()
        x = self.emb_layer(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(x_i,x_i.size(2)).squeeze(2) for x_i in x]
        x = torch.cat(x,1)
        x = self.dropout(x)

        x=F.relu(self.fc1(x))

        if post_types is not None:
            #print('----')
            x[post_types==0]=self.emb_layer(torch.tensor(1).to(x.device))
            x[post_types==1]=self.emb_layer(torch.tensor(2).to(x.device))
        #print(x.shape)
        return x

class attentive_pooling(torch.nn.Module):
    def __init__(self,emb,emb_size,ffn=False,hidden_size=100):
        super(attentive_pooling,self).__init__()
        self.emb_layer = nn.Embedding(100000, 100, padding_idx=0, _weight=emb)
        self.afc1=nn.Linear(emb_size,int(emb_size/2))
        self.afc2=nn.Linear(int(emb_size/2),1)
        self.softmax=nn.Softmax()

    def forward(self,x):
        x=x.long()
        x_mask=x.bool()
        x = self.emb_layer(x)
        score = self.afc2(F.relu(self.afc1(x))).squeeze(-1)
        score[~x_mask]=1e-12
        att=self.softmax(score)
        x = torch.matmul(att.unsqueeze(1),x).squeeze(1)
        x = F.relu(x)
        return x

class Att_pool(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Att_pool,self).__init__()
        self.gate_nn = nn.Sequential(nn.Linear(hidden_size,hidden_size),
                                     nn.Tanh(),
                                     nn.Linear(hidden_size,1,bias=False))
    def forward(self,x):
        score = self.gate_nn(x).squeeze(-1)
        score = F.softmax(score).unsqueeze(-1)
        x = x*score
        x = x.sum(1).squeeze(1)
        return x



class word_pooling(torch.nn.Module):
    def __init__(self,emb,pool_type):
        super(word_pooling, self).__init__()
        self.pool_type=pool_type
        self.emb_layer = nn.Embedding(100000, 100, padding_idx=0, _weight=emb)
    def forward(self,x):
        x=x.long()
        #print(x.shape)
        x_len=x.bool().long().sum(-1).unsqueeze(-1).detach()
        x_len[x_len==0]=1
        #print(x_len.shape)
        x = self.emb_layer(x)
        #print(x.shape)
        if self.pool_type=='max':
            x = x.max(1)[0].squeeze()
        #    print(x.shape)
        elif self.pool_type=='mean':
            x = x.mean(1).squeeze()/x_len
        return x



def graph_pool(x, cluster, mode='mean'):
    if mode=='max':
        x_pool = scatter_max(x,cluster,dim=0)[0]
    else:# mode=='mean':
        x_pool = scatter_mean(x,cluster,dim=0)
    return x_pool





class Multi_View_Graph_model_with_cross_network(torch.nn.Module):
    def __init__(self,args,tweet_emb=None,user_emb=None):
        super(Multi_View_Graph_model_with_cross_network, self).__init__()

        print('=============')
        for key, val in vars(args).items():
            print(key,val)

        self.tweet_max_len = 50
        self.user_max_len = 50
        self.tweet_word_dic_size=100000
        self.user_word_dic_size=100000
        self.emb_size=100
        #self.hidden_size=args.hidden_size
        self.text_hidden_size=args.hidden_size

        self.meta_hidden_size=20


        text_feat_size_map={'w2v':100,'bert':768,'tfidf':5000,'wids-split':100,'wids-share':100}
        self.text_emb_size = text_feat_size_map[args.text_type]


        self.args=args


        self.drop = nn.Dropout(p=args.dropout_rate)  #
        self.relu = nn.ReLU()#nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.n_heads=2



        print('input_size',self.text_emb_size)


        if tweet_emb is not None:
            tweet_emb = torch.tensor(tweet_emb,dtype=torch.float)
            if args.text_encoder=='cnn':
                self.post_encoder = cnn_encoder(tweet_emb,self.emb_size,kernel_num=50)
            elif args.text_encoder=='lstm':
                self.post_encoder = lstm_encoder(tweet_emb,self.emb_size,hidden_size=100,n_layers=2)
            elif args.text_encoder=='attpool':
                self.post_encoder = attentive_pooling(tweet_emb,self.emb_size)
            elif args.text_encoder=='maxpool':
                self.post_encoder= word_pooling(tweet_emb,'max')
            elif args.text_encoder=='meanpool':
                self.post_encoder= word_pooling(tweet_emb,'mean')
            else:
                assert False

        else:
            self.post_encoder = lambda x: x


        if user_emb is not None:
            user_emb = torch.tensor(user_emb,dtype=torch.float)
            if args.text_encoder=='cnn':
                self.user_encoder = cnn_encoder(user_emb,self.emb_size,kernel_num=50)
            elif args.text_encoder=='lstm':
                self.user_encoder = lstm_encoder(user_emb,self.emb_size,hidden_size=self.emb_size,n_layers=2)
            elif args.text_encoder=='attpool':
                self.user_encoder = attentive_pooling(user_emb,self.emb_size)
            elif args.text_encoder == 'maxpool':
                self.user_encoder = word_pooling(user_emb, 'max')
            elif args.text_encoder == 'meanpool':
                self.user_encoder = word_pooling(user_emb, 'mean')
            else:
                assert False
        else:
            self.user_encoder=self.post_encoder


        self.post_meta_encoder = nn.Linear(args.post_meta_size,self.meta_hidden_size)
        self.user_meta_encoder = nn.Linear(args.user_meta_size,self.meta_hidden_size)

        #gnn layers
        #input_size_list=[self.text_emb_size,args.post_meta_size,self.text_emb_size,args.user_meta_size]
        #if args.bidirectional:



        #self.input_layer_list=torch.nn.ModuleList([nn.Linear(input_size_list[i],gnn_hidden_size_list[i]) for i in range(4)])
        self.p_layer = nn.Linear(self.text_emb_size+args.post_meta_size, args.hidden_size)
        self.u_layer = nn.Linear(self.text_emb_size+args.user_meta_size, args.hidden_size)


        if args.node_filter:
            self.p_filter_layer = nn.Linear(args.post_meta_size,1)
            self.u_filter_layer = nn.Linear(args.user_meta_size,1)



        gnn_hidden_size_list=[args.hidden_size,args.hidden_size]
        gnn_input_size_list=gnn_hidden_size_list

        #gnn
        self.gnn_list=[]
        input_size = gnn_input_size_list[0]
        hidden_size= gnn_hidden_size_list[0]

        self.gnn_list.append(torch.nn.ModuleList([TreeGATConv(input_size,int(hidden_size/self.n_heads),
                                                          heads=self.n_heads,dropout=0,add_self_loops=False),
                                                  TreeGATConv(hidden_size, int(hidden_size / self.n_heads),
                                                          heads=self.n_heads, dropout=0,add_self_loops=False)
                                                  ]))

        input_size = gnn_input_size_list[1]
        hidden_size=gnn_hidden_size_list[1]
        print('rgat')
        self.gnn_list.append(torch.nn.ModuleList([RGATConv(input_size,int(hidden_size/self.n_heads),add_self_loops=False,
                                                          heads=self.n_heads,dropout=0),
                                                  RGATConv(hidden_size, int(hidden_size / self.n_heads),add_self_loops=False,
                                                          heads=self.n_heads, dropout=0)
                                                  ]))




        self.gnn_list=torch.nn.ModuleList(self.gnn_list)



        self.post_map = nn.Linear(args.hidden_size,args.hidden_size)
        self.user_map = nn.Linear(args.hidden_size,args.hidden_size)
        if args.network_type == 'gatv2':
            self.fuse_layers = torch.nn.ModuleList([GATv2Conv(args.hidden_size,int(args.hidden_size/self.n_heads),
                                                                  heads=self.n_heads,dropout=0) for _ in range(args.n_cross_layer)
                                                          ])
        else:
            raise ValueError('combine gnn type error')
        out_feat_size = args.hidden_size



        #gnn pool layers
        self.attpool_list=[]
        gnn_hidden_size_list = [args.hidden_size,args.hidden_size,args.hidden_size,args.hidden_size]
        for i in range(len(gnn_hidden_size_list)):
            hidden_size = gnn_hidden_size_list[i]
            gate_nn=nn.Sequential(nn.Linear(hidden_size,hidden_size),
                                  nn.Tanh(),
                                  nn.Linear(hidden_size,1,bias=False))
            self.attpool_list.append(GlobalAttention(gate_nn))
        self.attpool_list=torch.nn.ModuleList(self.attpool_list)



        #classifier layer
        self.veracity_fc1 = nn.Linear(out_feat_size, 100)
        self.veracity_fc2 = nn.Linear(100, 1)
        # GRL layer , topic classifier
        self.revgrad=GradientReversal(alpha=args.alpha)
        self.topic_fc1=nn.Linear(out_feat_size,100)
        self.topic_fc2=nn.Linear(100,4)

        if 'joint' in args.model_name:
            # self.joint_gnn=GATv2Conv(out_feat_size,out_feat_size)
            self.joint_gnn = SAGEConv(out_feat_size, out_feat_size)
            self.joint_fc = nn.Linear(out_feat_size,1)


    def forward(self, batch, extract_feat=False):
        posts, users=batch[:2]
        u2p_edges,u2rp_edges = batch[-2:]


        pt,pm, p_edges, p_cluster, post_types,puids,p2p_edge_dist = posts.t,posts.m,posts.edge_index,posts.batch,posts.post_types,posts.puids,posts.edge_dist
        ut,um, u_edges, u_cluster = users.t,users.m,users.edge_index,users.batch



        pt = self.post_encoder(pt)
        ut = self.user_encoder(ut)

        if self.args.node_filter:
            p_filter = F.sigmoid(self.p_filter_layer(pm))
            u_filter = F.sigmoid(self.u_filter_layer(um))
            pt = pt * p_filter
            ut = ut * u_filter



        p = torch.cat([pt,pm],dim=-1)
        p = self.p_layer(p)
        u = torch.cat([ut,um],dim=-1)
        u = self.u_layer(u)


        if self.training:#add noise
            if p_edges.shape[0]>0 and p_edges.shape[1]>10:
                p_edge_mask=torch.randint(100,size=(p_edges.shape[1],))<int(100*self.args.edge_drop_rate)
                p_edges = p_edges.T[~p_edge_mask].T
            if u_edges.shape[0]>0 and u_edges.shape[1]>10:
                u_edge_mask=torch.randint(100,size=(u_edges.shape[1],))<int(100*self.args.edge_drop_rate)
                u_edges=u_edges.T[~u_edge_mask].T



        x_list = [p,u]
        edges_list = [p_edges,u_edges]
        mid_list = []


        #post-post tree



        for i,layers in enumerate(self.gnn_list):
            x = x_list[i]

            if i==1:#relational user graph
                edges_f = edges_list[i]
                if edges_f.shape[0]>0:
                    n_edges =edges_f.shape[1]
                    edges_f = add_self_loops(edges_f,num_nodes=x.shape[0])[0]
                    edges_r = torch.cat([edges_f[1].reshape(1,-1),edges_f[0].reshape(1,-1)],dim=0)
                    edges = torch.cat([edges_f,edges_r],dim=-1)
                    edge_values = torch.cat([torch.ones(edges_f.shape[-1],device=edges.device),2*torch.ones(edges_r.shape[-1],device=edges.device)])
                    edges, edge_values= coalesce(edges, edge_values, m=x.shape[0], n=x.shape[0])
                    edge_type = edge_values-1
                    #edge_type = torch.tensor([0]*n_edges+[1]*n_edges,device=x.device)

                else:
                    edges=edges_f
                    edges_r=edges_f
                    n_edges=0
                    edge_type = torch.tensor([])

            else:
                if edges_list[i].shape[0]>0:
                    edges,p2p_edge_dist = to_undirected(edges_list[i],p2p_edge_dist)

            for layer in layers:
                res = x.clone()
                if i==1:#relational user graph
                    x = layer(x, edges,edge_type)
                else:
                    x = layer(x, edges,edge_type=p2p_edge_dist)
                x+=res
                x = self.drop(self.relu(x))

            mid_list.append(x)





        p,u = mid_list
        p = self.post_map(p)
        u = self.user_map(u)
        #如何拼起来
        x_list=[]
        x_cluster_list=[]
        x_in_p_list = []

        for i in range(p_cluster.max()+1):
            x_list.append(p[p_cluster == i])
            x_in_p_list.append(torch.ones(x_list[-1].shape))
            x_list.append(u[u_cluster == i])
            x_in_p_list.append(torch.zeros(x_list[-1].shape))
            x_cluster_list.append(p_cluster[p_cluster==i])
            x_cluster_list.append(u_cluster[u_cluster==i])

        x = torch.cat(x_list,dim=0)
        x_cluster = torch.cat(x_cluster_list)
        x_in_p = torch.cat(x_in_p_list).bool()

        u2p_edges = to_undirected(u2p_edges)
        u2p_edges = add_self_loops(u2p_edges,num_nodes=p.shape[0]+u.shape[0])[0]

        for i in range(self.args.n_cross_layer):
            res=x.clone()
            conv=self.fuse_layers[i]
            x = conv(x,u2p_edges)
            x+=res
            x = self.drop(self.relu(x))


        x_p_fused = x[x_in_p]
        x_u_fused = x[~x_in_p]

        x_p = p+x_p_fused
        x_u = u+x_u_fused
        cluster_list=[p_cluster,u_cluster]


        #two view pooling
        out_list = []
        for i, x in enumerate([x_p,x_u]):
            if self.args.pool_type in ('mean', 'max'):
                x = graph_pool(x, cluster_list[i], self.args.pool_type)
            elif self.args.pool_type == 'att':
                x = self.attpool_list[i](x, cluster_list[i])
            out_list.append(x)#.unsqueeze(1))  # batch_size,hidden_size



        feats = torch.cat(out_list, dim=-1).squeeze(1)
        out = self.veracity_fc2(self.drop(self.relu(self.veracity_fc1(feats))))
        topic_out=self.topic_fc2(self.drop(self.relu(self.topic_fc1(self.revgrad(feats)))))



        all_out=None


        if extract_feat:
            return out.reshape(-1),topic_out,all_out,feats
        else:
            return out.reshape(-1),topic_out,all_out