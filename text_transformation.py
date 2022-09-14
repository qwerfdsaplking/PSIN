from transformers import AutoModel
from trash.load_database import *
import h5py
import warnings
import pickle

import torch
warnings.filterwarnings('ignore')
import numpy as np
#from tqdm.auto import tqdm
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
from TweetNormalizer import *
from gensim import models
from torch_geometric.utils import subgraph

def vec2str(vec):
    vec = ','.join([str(x) for x in vec])
    return vec

def is_token_valid(token):
    if token in tfidf_stop_words:
        return False
    if len(token) == 1 and token not in 'qwertyuiopasdfghjklzxcvbnm?!':
        return False
    return True


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


def tweet_normalize_and_tokenize(text):
    norm_text = normalizeTweet(text)
    norm_tokens = [normalizeToken(token).replace('#', '').lower() for token in norm_text.split(' ')]
    norm_tokens = [token for token in norm_tokens if len(token) > 0]
    return norm_tokens



def read_glove_embedding():
    path = './glove.twitter.27B.100d.txt'
    print('loading %s'%path)
    f = open(path,'r')
    word_dict={}
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.split(' ')
        word_dict[line[0]]=np.array(line[1:],dtype=np.float)
        assert len(line)==101
    word_dict['httpurl']=word_dict['<url>']
    word_dict['@user']=word_dict['<user>']
    return word_dict


def get_tweet_ids(data_list):
    tid_list = []
    for data in tqdm(data_list):
        tweets = json.loads(data['tweets'])
        replies = json.loads(data['replies'])
        retweets = json.loads(data['retweets'])
        posts = tweets + retweets + replies
        pids = [p['id'] for p in posts]
        tid_list.extend(pids)
    print(len(tid_list))
    print(len(set(tid_list)))


def get_word_embedding(tokens, embs, mode='mean'):
    if len(tokens) == 0:
        return [0.0] * 100  # vec2str([0.0]*300)

    vectors = [embs[key].reshape(1, -1) if key in embs else np.zeros([1, 100]) for key in tokens]
    matrix = np.concatenate(vectors, 0)
    if mode == 'mean':
        ave_feats = matrix.mean(0).reshape(-1).tolist()
    elif mode == 'max':
        ave_feats = matrix.max(0).reshape(-1).tolist()
    else:
        raise ValueError('Mode Error')

    # ave_feats = vec2str(ave_feats)
    return ave_feats


def get_bert_embedding(text, mode='mean'):
    # if len(set('qwertyuioplkjhgfdsazxcvbnm') & set(text))==0:
    #    text=''
    input_ids = torch.tensor([bertweet_tokenizer.encode(text)]).to(device)
    with torch.no_grad():
        features = bertweet(input_ids)[0]

    if mode == 'mean':
        ave_feats = features.mean(1).squeeze().tolist()
    elif mode == 'max':
        ave_feats = features.max(1).squeeze().tolist()
    else:
        raise ValueError('Mode Error')

    # ave_feats = vec2str(ave_feats)
    return ave_feats


def get_bert_embedding_batch(texts_batch, mode='mean'):
    # for i in range(len(texts_batch)):
    #    if len(set('qwertyuioplkjhgfdsazxcvbnm') & set(texts_batch[i]))==0:
    #        texts_batch[i]=''
    # bertweet.cpu()
    # input_ids_batch = torch.tensor([bertweet_tokenizer.batch_encode_plus(texts_batch)]).to(device)
    # batch = {k: torch.tensor(v).to(device) for k, v in bertweet_tokenizer.batch_encode_plus(texts_batch,padding=True).items()}
    batch = torch.tensor(
        bertweet_tokenizer.batch_encode_plus(texts_batch, padding=True, truncation=True, max_length=128)[
            'input_ids']).to(device)
    with torch.no_grad():
        features = bertweet(batch)[0]

    if mode == 'mean':
        ave_feats = features.mean(1).squeeze(1).tolist()  # numpy()#tolist()
    elif mode == 'max':
        ave_feats = features.max(1).squeeze(1).tolist()  # numpy()#tolist()
    else:
        raise ValueError('Mode Error')

    # ave_feats = vec2str(ave_feats)
    return ave_feats


def fit_tfidf(tokens_list, n_max=5000):
    token_dict = {}
    for tokens in tqdm(tokens_list):
        tokens = [tk for tk in tokens if is_token_valid(tk)]
        for tk in set(tokens):
            if tk not in token_dict:
                token_dict[tk] = 1
            else:
                token_dict[tk] += 1

    n_texts = len(tokens_list)

    idf_dict = {}
    for k, v in token_dict.items():
        idf_dict[k] = (1 + n_texts) / (1 + v)

    topn_idfs = sorted(idf_dict.items(), key=lambda item: item[1], reverse=False)[:n_max]

    topn_idf_dict = {}
    for k, v in topn_idfs:
        topn_idf_dict[k] = v

    word_id_dict = {}
    for i, k in enumerate(topn_idfs):
        word_id_dict[k[0]] = i
    return word_id_dict, topn_idf_dict


def get_tfidf_embedding(tokens, word_id_dict, topn_idf_dict):
    tokens = [tk for tk in tokens if is_token_valid(tk)]
    token_n_dict = {}
    for tk in tokens:
        if tk in token_n_dict:
            token_n_dict[tk] += 1
        else:
            token_n_dict[tk] = 1
    # tfidf_vec=[0]*n_max
    tfidf_sparse_vec = []
    for tk, tf in token_n_dict.items():
        if tk in word_id_dict:
            # tfidf_vec[word_id_dict[tk]]=topn_idf_dict[tk]
            tfidf_sparse_vec.extend([word_id_dict[tk], tf / topn_idf_dict[tk]])

    return  tfidf_sparse_vec# tfidf_vec#tfidf_sparse_vec  vec2str(tfidf_sparse_vec)





if mode=='bert':
    for i in range(10):
        try:
            bertweet_tokenizer = AutoTokenizer.from_pretrained('./bertweet_tokenizer', use_fast=False,normalization=True)#'vinai/bertweet-base'
            bertweet = AutoModel.from_pretrained('./bertweet')#('vinai/bertweet-base')

            break
        except Exception as e:
            print(i,e)
            pass
    bertweet.to(device)



re_tfidf=True
batch_size=8
text_len=50
tfidf_len=5000


def parallel_data_processor(data_list, func, n_workers=16,batch_size=128):
    data_loader = DataLoader(data_list,collate_fn=func,num_workers=n_workers,shuffle=False,batch_size=batch_size)
    processed_data_list=[]
    for batch in data_loader:
        processed_data_list.extend(batch)



def collate_fn(batch):
    batch = [tweet_normalize_and_tokenize(text) for text in batch]
    for i in range(len(batch)):
        if len(batch[i])>text_len:
            text = ' '.join(batch[i])
            while '@user @user @user @user @user' in text:
                text = text.replace('@user @user @user @user @user','@user')
            batch[i]=text.split(' ')
    return [tk[:text_len] for tk in batch]


mode=sys.argv[1]#'context'
if __name__=='__main__':
    print('load pkl')
    f = open('./datasets/text-meta.pkl', 'rb')
    post_texts, post_types, post_ids, post_metas, user_texts, user_ids, user_metas = pickle.load(f)
    f.close()

    all_texts = post_texts + user_texts
    #all_tokens = [tweet_normalize_and_tokenize(text) for text in tqdm(all_texts)]

    if mode in ['tfidf','w2v','bert','word_ids']:
        if mode not in ['user_meta','edges']:
            all_tokens=[]
            text_loader = DataLoader(all_texts,collate_fn=collate_fn,num_workers=16,shuffle=False,batch_size=4096)
            for tokens in tqdm(text_loader):
                all_tokens.extend(tokens)

            post_tokens = all_tokens[:len(post_ids)]
            user_tokens = all_tokens[len(post_ids):]


            token_lens = pd.Series([len(tk) for tk in all_tokens])
            print(token_lens.describe())


        dump_pkl([post_ids,user_ids,post_types],'./datasets/ids.pkl')


    #title_df = pd.read_csv('./datasets/titles.csv')
    #titles=title_df['title'].tolist()
    #title_tokens = [tweet_normalize_and_tokenize(text) for text in titles]


    print('======')



    if mode=='tfidf':
        if os.path.exists('tfidf.pkl') and not re_tfidf:
            f = open('tfidf.pkl', 'rb')
            word_id_dict, topn_idf_dict=pickle.load(f)
            f.close()
        else:
            #total_tokens_set = [tweet_normalize_and_tokenize(text) for text in tqdm(all_tokens)]
            word_id_dict, topn_idf_dict = fit_tfidf(all_tokens,n_max=tfidf_len)
            f=open('tfidf.pkl','wb')
            pickle.dump([word_id_dict, topn_idf_dict],f)
            f.close()

        tfidf_vecs = [get_tfidf_embedding(tokens, word_id_dict, topn_idf_dict) for tokens in all_tokens]
        posts_tfidf_vecs = tfidf_vecs[:len(post_ids)]
        users_tfidf_vecs = tfidf_vecs[len(post_ids):]
        #title_vecs = [get_tfidf_embedding(tokens, word_id_dict, topn_idf_dict) for tokens in title_tokens]

        dump_pkl(posts_tfidf_vecs,'./datasets/tweet_tfidf_feat.pkl')
        dump_pkl(users_tfidf_vecs,'./datasets/user_tfidf_feat.pkl')



    if mode == 'w2v':
        #vector_path = './GoogleNews-vectors-negative300.bin'
        #google_emb = models.KeyedVectors.load_word2vec_format(vector_path, binary=True)
        glove_emb = read_glove_embedding()
        word_vecs = [get_word_embedding(tokens, glove_emb, mode='mean') for tokens in tqdm(all_tokens)]
        posts_word_vecs = word_vecs[:len(post_ids)]
        users_word_vecs = word_vecs[len(post_ids):]
        dump_npy(np.array(posts_word_vecs,dtype=np.float),'./datasets/tweet_w2v_feat.npy')
        dump_npy(np.array(users_word_vecs,dtype=np.float),'./datasets/user_w2v_feat.npy')



    if mode == 'bert':
        batch_size=256
        bertweet_tokenizer = AutoTokenizer.from_pretrained('./bertweet_tokenizer', use_fast=False,                                                       normalization=True)  # 'vinai/bertweet-base'
        bertweet = AutoModel.from_pretrained('./bertweet')  # ('vinai/bertweet-base')

        bertweet.to(device)
        bert_vecs = []
        for i in tqdm(range(0, len(all_texts), batch_size)):
            texts_batch = all_texts[i:i + batch_size]
            bert_vecs_batch = get_bert_embedding_batch(texts_batch, mode='mean')
            bert_vecs.extend(bert_vecs_batch)
        posts_bert_vecs = bert_vecs[:len(post_ids)]
        users_bert_vecs = bert_vecs[len(post_ids):]
        dump_npy(np.array(posts_bert_vecs,dtype=np.float),'./datasets/tweet_bert_feat.npy')
        dump_npy(np.array(users_bert_vecs,dtype=np.float),'./datasets/user_bert_feat.npy')




    if mode=='word_ids':
        #vector_path = './GoogleNews-vectors-negative300.bin'
        #google_emb = models.KeyedVectors.load_word2vec_format(vector_path, binary=True)
        glove_emb = read_glove_embedding()
        mprint(len(post_texts))


        # 不是reply全部替换成空字符串
        all_post_texts = [text if post_types[i] == 3 else '' for i, text in enumerate(post_texts)]
        mprint(len(all_post_texts))


        def texts2ids(all_tokens):  # tweet和user不一样。。。
            token_dict = {}
            for tokens in all_tokens:
                for tk in tokens:
                    if tk not in token_dict:
                        token_dict[tk] = 1
                    else:
                        token_dict[tk] += 1

            token_freq_list = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)
            mprint(len(token_freq_list))
            filtered_token_freq_list = token_freq_list[:100000-6]
            mprint(len(filtered_token_freq_list))


            token_id_dict = {'[#PAD#]': 0, '[#TWEET#]': 1, '[#RETWEET#]': 2, '[#REPLY#]': 3,'[#USER#]': 4, '[#OOV#]': 5}

            for tk in filtered_token_freq_list:
                token_id_dict[tk[0]] = len(token_id_dict)

            def map_token2id(tokens):
                ids = [token_id_dict[tk] if tk in token_id_dict else token_id_dict['[#OOV#]'] for tk in tokens]
                ids += [0]*(text_len-len(ids))
                return ids

            all_ids = [map_token2id(tokens) for i, tokens in tqdm(enumerate(all_tokens))]


            return all_ids, filtered_token_freq_list, token_id_dict


        #mode_v2 = 'share'


        def get_sub_embedding(word_id_dict,n_words,vec_size):
            word_embeddings = np.random.randn(n_words, vec_size)
            for word, tid in word_id_dict.items():
                if word in glove_emb:
                    word_embeddings[tid]=np.array(glove_emb[word])
                #else:
                #    word_embeddings[tid]=np.zeros(vec_size)
            word_embeddings[0]=np.zeros(vec_size)
            return word_embeddings

    #if mode_v2 == 'split':
        post_token_ids, filtered_post_token_freq_list, post_token_id_dict = texts2ids(post_tokens)
        user_token_ids, filtered_user_token_freq_list, user_token_id_dict = texts2ids(user_tokens)
        post_word_embeddings = get_sub_embedding(post_token_id_dict,100000,100)
        user_word_embeddings = get_sub_embedding(user_token_id_dict,100000,100)
        dump_npy(post_word_embeddings,'./datasets/split_post_embedding.npy')
        dump_npy(user_word_embeddings,'./datasets/split_user_embedding.npy')
        dump_npy(np.array(post_token_ids,dtype=np.long),'./datasets/split_post_token_ids.npy')
        dump_npy(np.array(user_token_ids,dtype=np.long),'./datasets/split_user_token_ids.npy')


    #elif mode_v2 == 'share':
        all_token_ids, filtered_token_freq_list,token_id_dict = texts2ids(all_tokens)
        post_token_ids = all_token_ids[:len(post_ids)]
        user_token_ids = all_token_ids[len(post_ids):]
        word_embeddings = get_sub_embedding(token_id_dict,100000,100)
        dump_npy(word_embeddings,'./datasets/share_all_embedding.npy')
        dump_npy(np.array(post_token_ids,dtype=np.long),'./datasets/share_post_token_ids.npy')
        dump_npy(np.array(user_token_ids,dtype=np.long),'./datasets/share_user_token_ids.npy')



    if mode=='meta':
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


        #def meta_normalize(metas):
        #    metas = np.array(metas)
        #    metas = (metas - metas.mean(0)) / metas.std(0)
        #    return metas.tolist()

        # all_post_metas = meta_normalize(all_post_metas)
        # all_user_metas = meta_normalize(all_user_metas)

        # sentiment
        analyzer = SentimentIntensityAnalyzer()
        all_post_sentiment = []
        for i, ptype in tqdm(enumerate(post_types)):
            if ptype == 3:
                text = post_texts[i]
                vs = analyzer.polarity_scores(text)
                neg, neu, pos, compound = vs['neg'], vs['neu'], vs['pos'], vs['compound']
            else:
                compound = 1
            all_post_sentiment.append(compound)

        post_metas = [meta+tuple([all_post_sentiment[i]]) for i,meta in enumerate(post_metas)]

        dump_npy(np.array(post_metas,dtype=np.float),'./datasets/post_metas.npy')
        dump_npy(np.array(user_metas,dtype=np.float),'./datasets/user_metas.npy')



    if mode=='politics_words':
        print('processing context')
        data_df = pd.read_csv('./datasets/final_dataset.csv')
        titles = data_df[data_df['data_name_v2']=='politifact-1']['title'].values.tolist()

        all_tokens = []
        text_loader = DataLoader(titles, collate_fn=collate_fn, num_workers=16, shuffle=False, batch_size=4096)
        for tokens in tqdm(text_loader):
            all_tokens.extend(tokens)

        _, topn_idf_dict = fit_tfidf(all_tokens, n_max=1000)
        topn_idf = sorted(topn_idf_dict.items(), key=lambda item: item[1], reverse=False)
        topn_val = [(x[0], len(all_tokens) / x[1]) for x in topn_idf]

    if mode=='word_cloud':
        all_tokens = []
        text_loader = DataLoader(all_texts, collate_fn=collate_fn, num_workers=16, shuffle=False, batch_size=4096)
        for tokens in tqdm(text_loader):
            all_tokens.extend(tokens)

        post_tokens = all_tokens[:len(post_ids)]
        user_tokens = all_tokens[len(post_ids):]

        post_labels=np.zeros(len(post_metas))
        user_fake_n=np.zeros(len(user_metas))
        user_real_n=np.zeros(len(user_metas))

        print('loading processed dataset')
        f = open('./datasets/processed_context_data_v2.pkl', 'rb')
        p_data_list = pickle.load(f)
        for i, inst in enumerate(p_data_list):
            news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, p_user_relations, data_name, data_name_v2, data_name_combined, data_name_combined_v2 = inst


            for pid in p_post_ids:
                if post_labels[pid]==0:
                    post_labels[pid]=labels


            if labels==1:
                for uid in p_user_ids:
                    user_fake_n[uid]+=1
            else:
                for uid in p_user_ids:
                    user_real_n[uid]+=1


        fake_post_tokens = [post_tokens[i]  for i in range(len(post_tokens))  if post_labels[i]==1]
        real_post_tokens = [post_tokens[i]  for i in range(len(post_tokens))  if post_labels[i]==0]

        fake_user_tokens = [user_tokens[i] for i in range(len(user_tokens)) if (user_fake_n[i]>2)]
        real_user_tokens = [user_tokens[i] for i in range(len(user_tokens)) if (user_fake_n[i]==0) and (user_real_n[i]>2)]


        topn_idf_list=[]

        for tokens_list in [fake_post_tokens,real_post_tokens,fake_user_tokens,real_user_tokens]:
            _,topn_idf_dict = fit_tfidf(tokens_list,n_max=1000)
            topn_idf = sorted(topn_idf_dict.items(), key=lambda item: item[1], reverse=False)
            topn_val = [(x[0],len(tokens_list)/x[1]) for x in topn_idf]
            topn_idf_list.append(topn_val)

        def filter_words(w):
            if w in ['httpurl','rt','not']:
                return False
            if len(w)==1:
                return False
            for c in w:
                if c not in list('qwertyuioplkjhgfdsazxcvbnm1234567890'):
                    return False
            return True


        topn_idf_filtered_list=[]
        for i,topn_idf in enumerate(topn_idf_list):
            topn_idf_filtered = [x for x in topn_idf if filter_words(x[0])]
            topn_idf_filtered_list.append(topn_idf_filtered)

            f=open('./cloud_%s.csv'%i,'w')
            for word,weight in topn_idf_filtered:
                f.write('%s,%s\n'%(int(weight),word))
            f.close()


    if mode=='user_cal':
        user_metas = load_npy('./datasets/user_metas.npy')
        user_fake_n=np.zeros(user_metas.shape[0])
        user_real_n = np.zeros(user_metas.shape[0])

        #user_verified=user_metas[:,1]
        #user_time = user_metas[:,6]

        user_metas=user_metas[:,1:]


        print('loading processed dataset')
        f = open('./datasets/processed_context_data_v2.pkl', 'rb')
        p_data_list = pickle.load(f)
        for i, inst in enumerate(p_data_list):
            news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, p_user_relations, data_name, data_name_v2, data_name_combined, data_name_combined_v2 = inst
            if labels==1:
                for uid in p_user_ids:
                    user_fake_n[uid]+=1
            else:
                for uid in p_user_ids:
                    user_real_n[uid]+=1

        #user_metas = np.concatenate([user_verified.reshape(-1,1),user_time.reshape(-1,1),user_fake_n.reshape(-1,1),user_real_n.reshape(-1,1)],axis=-1)
        user_metas = np.concatenate([user_metas,user_fake_n.reshape(-1,1),user_real_n.reshape(-1,1)],axis=-1)
        user_metas = pd.DataFrame(user_metas,columns=['verified','n_followers','n_followings','n_tweet','n_list','time','n_fake','n_real'])

        print(user_metas[(user_metas['n_fake']==0)&(user_metas['n_real']>3)].mean())
        print(user_metas[(user_metas['n_fake']>3)].mean())


        user_metas['n_followers_2']=0
        user_metas['n_followings_2']=0

        f = open('./datasets/processed_context_data_v2.pkl', 'rb')
        p_data_list = pickle.load(f)



        mprint('loading processed_edges.pkl')
        p_edges = load_npy('./datasets/processed_edges.npy')
        p_edges = pd.DataFrame(p_edges,columns=['src','dst'])
        p_edges['cnt']=1
        p_edges_src = p_edges.groupby('src')['cnt'].agg('sum')

        p_edges_dst = p_edges.groupby('dst')['cnt'].agg('sum')

        user_metas['n_followings_2'][p_edges_src.index]=p_edges_src
        user_metas['n_followers_2'][p_edges_dst.index]=p_edges_dst

        print(user_metas[(user_metas['n_fake']==0)&(user_metas['n_real']>3)].mean())
        print(user_metas[(user_metas['n_fake']>0)].mean())



    if mode=='sentiment_cal':

        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        all_post_sentiment = []
        for i, ptype in tqdm(enumerate(post_types)):
            if ptype == 3:
                text = post_texts[i]
                vs = analyzer.polarity_scores(text)
                neg, neu, pos, compound = vs['neg'], vs['neu'], vs['pos'], vs['compound']
            else:
                compound = 1
                neg=0
                neu=0
                pos=1
            all_post_sentiment.append([neg, neu, pos, compound])

        senti_meta = np.array(all_post_sentiment)


        print('loading processed dataset')
        f = open('./datasets/processed_context_data_v2.pkl', 'rb')
        p_data_list = pickle.load(f)

        post_metas = load_npy('./datasets/post_metas.npy')
        user_metas = load_npy('./datasets/user_metas.npy')

        data_list=[]

        for i,inst in enumerate(p_data_list):
            news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, p_user_relations, data_name, data_name_v2, data_name_combined, data_name_combined_v2 = inst

            senti_meta_feats = senti_meta[p_post_ids]  # 50/100/768

            post_types=np.array(post_types)
            reply_meta = senti_meta_feats[post_types==2]

            neg, neu, pos, compound = reply_meta.mean(0).tolist()

            data_list.append([news_id,labels,neg, neu, pos, compound])

        data_df = pd.DataFrame(data_list,columns=['news_id','labels','neg','neu','pos','compound'])





    if mode=='multi-hop':
        error_cnt=0

        f = open('./datasets/processed_context_data_v2.pkl', 'rb')
        p_data_list = pickle.load(f)
        p_data_list_v3=[]
        for inst in tqdm(p_data_list):
            news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, p_user_relations, data_name, data_name_v2, data_name_combined, data_name_combined_v2 = inst

            #p_post_ids = list(set(p_post_ids))
            #去重
            p_post_ids_set=set([])
            p_post_ids_v2=[]
            p_aligned_user_ids_v2=[]
            post_types_v2=[]
            for i,pid in enumerate(p_post_ids):
                if pid not in p_post_ids_set:
                    p_post_ids_v2.append(pid)
                    p_aligned_user_ids_v2.append(p_aligned_user_ids[i])
                    post_types_v2.append(post_types[i])
                    p_post_ids_set.add(pid)

            if len(p_post_ids_v2)<len(p_post_ids):
                error_cnt+=1
                print(error_cnt,news_id,len(p_post_ids_v2),len(p_post_ids))

            p_post_ids=p_post_ids_v2
            p_aligned_user_ids=p_aligned_user_ids_v2
            post_types=post_types_v2


            local_pid_map={}
            local_uid_map={}
            local_pid_map_r={}
            local_uid_map_r={}

            max_post_cnt=9000
            p_post_ids=p_post_ids[:max_post_cnt]

            for i,pid in enumerate(p_post_ids[:max_post_cnt]):
                local_pid_map[pid]=i
                local_pid_map_r[i]=pid
            for i,uid in enumerate(p_user_ids):
                local_uid_map[uid]=i
                local_uid_map_r[i]=uid

            post_ids_set=set(p_post_ids)

            p_post_edges = p_retweet_relations+p_reply_relations
            local_post_edges = [(local_pid_map[r[0]],local_pid_map[r[1]]) for r in p_post_edges if (r[0] in post_ids_set) and (r[1] in post_ids_set)]
            local_user_edges = [(local_uid_map[r[0]],local_pid_map[r[1]]) for r in p_write_relations if (r[1] in post_ids_set)]
            if len(p_post_ids)>300:
                device = torch.device('cuda', hvd_rank)
            else:
                device = torch.device('cpu')

            p2p_adj=torch.zeros(len(p_post_ids),len(p_post_ids),device=device)
            u2p_adj=torch.zeros(len(p_user_ids),len(p_post_ids),device=device)


            p2p_adj[[x[0] for x in local_post_edges],[x[1] for x in local_post_edges]]=1
            u2p_adj[[x[0] for x in local_user_edges],[x[1] for x in local_user_edges]]=1
            p2p_adj_all=torch.zeros(len(p_post_ids),len(p_post_ids),device=device)+p2p_adj

            #a=input('===')

            hop_i_adj = p2p_adj
            for i in range(20):
                if hop_i_adj.sum()==0:
                    break
                hop_i_adj=(hop_i_adj@p2p_adj).bool().float()*(i+2)
                p2p_adj_all+=hop_i_adj

            if p2p_adj_all.max()>21:
                print(p2p_adj_all.max())
                assert False

            u2p_adj_all = u2p_adj @ p2p_adj_all + u2p_adj

            p2p_adj_all=p2p_adj_all.long()#.bool().long()
            u2p_adj_all=u2p_adj_all.bool().long()

            p2p_adj_all=p2p_adj_all.cpu()
            p2p_adj=p2p_adj.cpu()
            u2p_adj=u2p_adj.cpu()

            n_post= p2p_adj_all.shape[0]
            p2p_adj_all[torch.arange(n_post),torch.arange(n_post)]=0
            p2p_edges_all=p2p_adj_all.nonzero().tolist()

            u2p_edges_all=u2p_adj_all.nonzero().tolist()

            p2p_edge_dist = [p2p_adj_all[x[0]][x[1]].item() for x in p2p_edges_all]

            p2p_edges_all = [(local_pid_map_r[r[0]],local_pid_map_r[r[1]]) for r in p2p_edges_all]
            u2p_edges_all = [(local_uid_map_r[r[0]],local_pid_map_r[r[1]]) for r in u2p_edges_all]


            if len(p_post_ids)<max_post_cnt:
                assert len(p_post_edges)<=len(p2p_edges_all)
                assert len(p_write_relations)<=len(u2p_edges_all)
            #print(i,len(p_post_edges),len(p2p_edges_all),len(p_write_relations),len(u2p_edges_all))

            #if len(p_post_edges)==len(p2p_edges_all):
            #    a=input('-----')

            inst = news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, p_user_relations,p2p_edges_all,u2p_edges_all,p2p_edge_dist, data_name, data_name_v2, data_name_combined, data_name_combined_v2

            p_data_list_v3.append(inst)

        dump_pkl(p_data_list_v3,'./datasets/processed_context_data_v4.pkl')





    if mode=='time':
        print('loading processed dataset')
        f = open('./datasets/processed_context_data.pkl', 'rb')
        p_data_list = pickle.load(f)

        post_ids, user_ids, post_types, post_metas, user_metas, posts_vecs, users_vecs, post_emb, user_emb = \
            load_post_user_feats_and_embeddings('tfidf')




        news_post_text_list=[]
        news_post_meta_list=[]
        news_user_text_list = []
        news_user_meta_list = []

        thresholds = [3600*(i+1) for i in range(10)]+[3600*24,3600*100,3600*10000]
        n_posts_list = [[] for _ in range(len(thresholds))]

        for inst in tqdm(p_data_list):
            news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, data_name, data_name_v2, data_name_combined, data_name_combined_v2 = inst
            times = post_metas[p_post_ids][:,-2]
            times = times-times[0]

            post_metas[p_post_ids,-2]=times

            for i,tds in enumerate(thresholds):
                n_posts_list[i].append((times<tds).sum())

        n_posts = np.array(n_posts_list)
        print(n_posts.mean(1))

    if mode=='context':

        pid2id_map={}
        for i,pid in enumerate(post_ids):
            pid2id_map[pid]=i
        uid2id_map={}
        for i,uid in enumerate(user_ids):
            uid2id_map[uid]=i



        print('processing context')
        data_df = pd.read_csv('./datasets/final_dataset.csv')#['news_id,labels,tweet_ids, retweet_ids, reply_ids, user_ids, retweet_relations, reply_relations, write_relations,data_name,data_name_v2,data_name_combined, data_name_combined_v2'.replace(' ','').split(',')]
        for col in ['tweet_ids','retweet_ids','reply_ids','user_ids']:
            data_df[col]=data_df[col].map(lambda x:[int(t) for t in str(x).split(',')] if x==x else [])
        for col in ['retweet_relations','reply_relations']:
            data_df[col]=data_df[col].map(lambda x:[(int(t.split('-')[0]),int(t.split('-')[1])) for t in str(x).split(',')] if (x==x) else [])

        all_posts = data_df[['tweets','retweets','replies']].values.tolist()
        all_write_relations=[]
        for tweets,retweets,replies in tqdm(all_posts):
            tweets=json.loads(tweets)
            retweets=json.loads(retweets)
            replies=json.loads(replies)
            posts=tweets+retweets+replies
            write_relations=[]
            for p in posts:
                tid=p['id']
                uid=p['author_id']
                write_relations.append((uid,tid))
            all_write_relations.append(write_relations)
        data_df['write_relations']=all_write_relations


        data_list = data_df['news_id,labels,tweet_ids,retweet_ids,reply_ids,user_ids,retweet_relations,reply_relations,write_relations,data_name,data_name_v2,data_name_combined,data_name_combined_v2'.split(',')].values.tolist()


        p_data_list=[]
        for data in tqdm(data_list):
            news_id,labels,tweet_ids, retweet_ids, reply_ids, user_ids, retweet_relations, reply_relations, write_relations,data_name,data_name_v2,data_name_combined, data_name_combined_v2 =data
            post_ids = tweet_ids+retweet_ids+reply_ids
            post_ids = list(set(post_ids))

            post_user_map={}

            for uid,tid in write_relations:
                post_user_map[tid]=uid

            post_ids.sort()

            #post_ids = post_ids[:max_n_posts]
            aligned_user_ids = [post_user_map[pid] for pid in post_ids]

            post_types = []
            for pid in post_ids:
                if pid in retweet_ids:
                    post_types.append(1)
                elif pid in reply_ids:
                    post_types.append(2)
                else:
                    post_types.append(0)


            root_pids = list(set(tweet_ids) | (set(retweet_ids)-set([r[0] for r in retweet_relations])))

            #map pid, uid to id
            p_post_ids = [pid2id_map[pid] for pid in post_ids]
            p_root_pids = [pid2id_map[pid] for pid in root_pids]
            p_user_ids = [uid2id_map[uid] for uid in user_ids]
            p_aligned_user_ids = [uid2id_map[uid] for uid in aligned_user_ids]
            p_retweet_relations = [(pid2id_map[r[0]],pid2id_map[r[1]]) for r in retweet_relations]
            p_reply_relations = [(pid2id_map[r[0]],pid2id_map[r[1]]) for r in reply_relations]
            p_write_relations = [(uid2id_map[r[0]],pid2id_map[r[1]]) for r in write_relations]

            p_data_list.append([news_id,labels,p_post_ids,p_user_ids,p_aligned_user_ids,p_root_pids, post_types,p_retweet_relations,p_reply_relations,p_write_relations,data_name,data_name_v2,data_name_combined,data_name_combined_v2])

        f=open('./datasets/processed_context_data.pkl','wb')
        pickle.dump(p_data_list,f)
        p_data_df = pd.DataFrame(p_data_list,columns='news_id,labels,post_ids,user_ids,aligned_user_ids,root_pids,post_types,retweet_relations,reply_relations,write_relations,data_name,data_name_v2,data_name_combined,data_name_combined_v2'.split(','))


    if mode=='edges':

        #edges
        mprint('over writing edges.pkl')
        f = open('./datasets/edges.txt','r')
        edges=[]
        for line in tqdm(f):
            if len(line)>1:
                x=line.split('-')
                edge=(int(x[0]),int(x[1]))
                edges.append(edge)

        print(len(edges))
        edges = list(set(edges))
        print(len(edges))
        dump_pkl(np.array(edges),'./datasets/edges.pkl')


        pid2id_map={}
        for i,pid in enumerate(post_ids):
            pid2id_map[pid]=i
        uid2id_map={}
        for i,uid in enumerate(user_ids):
            uid2id_map[uid]=i

        mprint('overwriting processed_edges.pkl')#12 minutes
        #edges = [(uid2id_map[x[0]],uid2id_map[x[1]]) for x in edges]
        p_edges = [(uid2id_map[x[0]],uid2id_map[x[1]]) for x in tqdm(edges) if x[0] in uid2id_map and x[1] in uid2id_map]
        #dump_pkl(p_edges,'./datasets/processed_edges.pkl')
        p_edges = np.array(p_edges,dtype=np.long)
        dump_npy(p_edges,'./datasets/processed_edges.npy')

    if mode == 'user_meta':

        p_edges = load_npy('./datasets/processed_edges.npy')
        edges_df = pd.DataFrame(p_edges, columns=['src_uid', 'dst_uid'])
        user_meta_df = pd.DataFrame(user_metas,
                                    columns=['description_len', 'verified', 'followers_counts', 'following_counts',
                                             'tweet_counts', 'listed_counts', 'user_created_times'])
        # user_meta_df=user_meta_df[['index','description_len','verified', 'followers_counts', 'following_counts', 'tweet_counts', 'listed_counts', 'user_created_times']]
        dst_meta_df = pd.merge(edges_df, user_meta_df, left_on='dst_uid', right_index=True, how='left')
        print(dst_meta_df.dropna(how='any').shape)
        dst_meta_df = dst_meta_df.drop(['dst_uid'], axis=1)
        print('merge finished')
        src_mean_df = dst_meta_df.groupby('src_uid').agg('mean')
        src_mean_df.columns = [x + '_mean_out' for x in src_mean_df.columns]
        print('mean finished')
        src_max_df = dst_meta_df.groupby('src_uid').agg('max')
        src_max_df.columns = [x + '_max_out' for x in src_max_df.columns]
        print('max finished')
        #src_min_df = dst_meta_df.groupby('src_uid').agg('min')
        #src_min_df.columns = [x + '_min_out' for x in src_min_df.columns]
        #print('min finished')
        src_std_df = dst_meta_df.groupby('src_uid').agg('std')
        src_std_df.columns = [x + '_std_out' for x in src_std_df.columns]
        print('std finished')
        src_df = pd.concat([src_mean_df, src_max_df, src_std_df], axis=1)
        src_df['index'] = src_df.index

        # concat

        src_meta_df = pd.merge(edges_df, user_meta_df, left_on='src_uid', right_index=True, how='left')
        print(src_meta_df.dropna(how='any'))
        src_meta_df = src_meta_df.drop(['src_uid'], axis=1)
        print('merge finished')
        dst_mean_df = src_meta_df.groupby('dst_uid').agg('mean')
        dst_mean_df.columns = [x + '_mean_in' for x in dst_mean_df.columns]
        print('mean finished')
        dst_max_df = src_meta_df.groupby('dst_uid').agg('max')
        dst_max_df.columns = [x + '_max_in' for x in dst_max_df.columns]
        print('max finished')
        #dst_min_df = src_meta_df.groupby('dst_uid').agg('min')
        #dst_min_df.columns = [x + '_min_in' for x in dst_min_df.columns]
        #print('min finished')
        dst_std_df = src_meta_df.groupby('dst_uid').agg('std')
        dst_std_df.columns = [x + '_std_in' for x in dst_std_df.columns]
        print('std finished')
        dst_df = pd.concat([dst_mean_df, dst_max_df, dst_std_df], axis=1)
        dst_df['index'] = dst_df.index

        user_meta_df['index'] = user_meta_df.index

        # concat
        # user_meta_df['index']=user_meta_df.index
        user_feat_df = pd.merge(user_meta_df, src_df, left_on='index', right_on='index', how='left')
        user_feat_df = pd.merge(user_feat_df, dst_df, left_on='index', right_on='index', how='left')
        user_feat_df = user_feat_df.drop(['index'], axis=1)
        user_feat_df = user_feat_df.fillna(user_feat_df.mean())
        #user_feat_mean = user_feat_df.mean(0)
        #user_feat_std = user_feat_df.std(0)
        #user_feat_df = (user_feat_df - user_feat_mean) / user_feat_std
        dump_npy(user_feat_df.values,'./datasets/processed_user_feats.npy')
        #user_feat_df.to_csv('./datasets/processed_node_feats.csv', index=False)



    if mode =='local_user_graph':
        from torch_sparse import SparseTensor
        def edges2sparsetensor(n_nodes, edge_index):
            value = torch.arange(edge_index.size(1))
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                 value=value,
                                 sparse_sizes=(n_nodes, n_nodes))#.t()
            adj_t.storage.rowptr()
            return adj_t
        print('generate local user graph')
        print('loading processed dataset')
        f = open('./datasets/processed_context_data.pkl', 'rb')
        p_data_list = pickle.load(f)
        p_edges = load_npy('./datasets/processed_edges.npy')
        edge_index = torch.tensor(p_edges).T
        adj = edges2sparsetensor(len(user_ids), edge_index)
        adj_t = adj.t()

        p_data_list_v2 = []
        for inst in tqdm(p_data_list):
            news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, data_name, data_name_v2, data_name_combined, data_name_combined_v2 = inst
            #x = input('xx')
            #p_user_ids.sort()
            #adj_m=subgraph(p_user_ids,edge_index)
            adj_m = adj[p_user_ids]
            adj_m_2=adj_m.t()[p_user_ids].t()

            row,col,_=adj_m_2.coo()
            local_edges = list(zip(row.tolist(),col.tolist()))
            p_user_edges = [(p_user_ids[x[0]],p_user_ids[x[1]]) for x in local_edges]
            inst = news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations,p_user_edges, data_name, data_name_v2, data_name_combined, data_name_combined_v2
            p_data_list_v2.append(inst)

        dump_pkl(p_data_list_v2,'./datasets/processed_context_data_v2.pkl')
            #adj_t_m = adj_t[p_user_ids]
            #x=input('xx')
            #local_adj, nids = adj_m.sample_adj(torch.tensor(p_user_ids), 5000, replace=False)
            #row, col, _ = adj_m.coo()
            #edges_m = torch.cat([row.reshape(1,-1),col.reshape(1,-1)])

            #adj_m = subgraph(p_user_ids, edge_index)
            #edges = list(zip(row,col))
            #row_t,col_t,_ = adj_t_m.coo()
            #edges_t = list(zip(col_t,row_t))

            #edges_f = set(edges)&set(edges_t)
            #row=row.tolist()
            #col=col.tolist()
            #row = [r for i,r in enumerate(row) if col[i] in p_user_ids]
            #col=[c for c in col if c in p_user_ids]
            #这个sample sample的是出度还是入度



    if mode=='gather_news_features':
       # from local_network_model import load_post_user_feats_and_embeddings
        print('loading processed dataset')
        f = open('./datasets/processed_context_data.pkl', 'rb')
        p_data_list = pickle.load(f)

        post_ids, user_ids, post_types, post_metas, user_metas, posts_vecs, users_vecs, post_emb, user_emb = \
            load_post_user_feats_and_embeddings('tfidf')




        news_post_text_list=[]
        news_post_meta_list=[]
        news_user_text_list = []
        news_user_meta_list = []
        for inst in tqdm(p_data_list):
            news_id, labels, p_post_ids, p_user_ids, p_aligned_user_ids, p_root_pids, post_types, p_retweet_relations, p_reply_relations, p_write_relations, data_name, data_name_v2, data_name_combined, data_name_combined_v2 = inst

            post_meta_feats = post_metas[p_post_ids]  # 50/100/768
            news_post_meta_mean = post_meta_feats.mean(0)
            news_post_meta_max=post_meta_feats.max(0)
            news_post_meta_std=post_meta_feats.std(0)
            news_post_meta_sum=post_meta_feats.sum(0)
            news_post_meta = np.concatenate([news_post_meta_sum,news_post_meta_mean,news_post_meta_max,news_post_meta_std])


            user_meta_feats = user_metas[p_user_ids]  # 50/100/768
            news_user_meta_sum = user_meta_feats.sum(0)
            news_user_meta_mean = user_meta_feats.mean(0)
            news_user_meta_max=user_meta_feats.max(0)
            news_user_meta_std=user_meta_feats.std(0)
            news_user_meta = np.concatenate([news_user_meta_sum,news_user_meta_mean,news_user_meta_max,news_user_meta_std])


            #user_meta_feats = user_metas[p_aligned_user_ids]  # 7?
            post_text_list=[]
            for i,pid in enumerate(p_post_ids):
                ks = posts_vecs[pid][::2]
                vs = posts_vecs[pid][1::2]
                post_text = np.zeros(5000)
                post_text[ks]=vs
                post_text_list.append(post_text.reshape(1,-1))
            news_post_text=np.concatenate(post_text_list,axis=0)
            news_post_text_mean=news_post_text.mean(0)
            #news_post_text_max=news_post_text.max(0)



            user_text_list=[]
            for i,uid in enumerate(p_aligned_user_ids):
                ks = users_vecs[uid][::2]
                vs = users_vecs[uid][1::2]
                user_text = np.zeros(5000)
                user_text[ks]=vs
                user_text_list.append(user_text.reshape(1,-1))
            news_user_text=np.concatenate(user_text_list,axis=0)
            news_user_text_mean=news_user_text.mean(0)
            #news_user_text_max=news_user_text.max(0)



            #news_text = np.concatenate([news_post_text_mean,news_user_text_mean]).reshape(1,-1)

            news_post_text_list.append(news_post_text_mean.reshape(1,-1))
            news_post_meta_list.append(news_post_meta.reshape(1,-1))
            news_user_text_list.append(news_user_text_mean.reshape(1, -1))
            news_user_meta_list.append(news_user_meta.reshape(1, -1))

            #gc.collect()


        news_post_texts = np.concatenate(news_post_text_list,axis=0)
        news_user_texts = np.concatenate(news_user_text_list, axis=0)

        news_post_metas = np.concatenate(news_post_meta_list, axis=0)
        news_user_metas = np.concatenate(news_user_meta_list, axis=0)

        dump_npy(news_post_texts,'./datasets/news_post_texts.npy')
        dump_npy(news_user_texts,'./datasets/news_user_texts.npy')

        dump_npy(news_post_metas,'./datasets/news_post_metas.npy')
        dump_npy(news_user_metas,'./datasets/news_user_metas.npy')




