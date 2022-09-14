from trash.load_database import *
import h5py
import warnings
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from TweetNormalizer import *

import torch

warnings.filterwarnings('ignore')
import numpy as np
# from tqdm.auto import tqdm
from tqdm import tqdm

from TweetNormalizer import *

if __name__ == '__main__':
    #os.chdir("/apdcephfs/private_erxuemin/erxuemin/Graph_fake_news")
    pass

def vec2str(vec):
    vec = ','.join([str(x) for x in vec])
    return vec


# tidif stop words, do not include query words such as where,which, and negative words
def drop_tables(cur):
    cur.execute('drop table users')
    cur.execute('drop table posts')
    conn.commit()


def create_tables(cur):
    try:
        cur.execute('''Create table posts
                    (id TEXT PRIMARY KEY NOT NULL,
                    tfidf_vec TEXT NOT NULL,
                    w2v_vec TEXT NOT NULL,
                    bert_vec TEXT NOT NULL,
                    retweet_cnt INT NOT NULL,
                    reply_cnt INT NOT NULL,
                    like_cnt INT NOT NULL,
                    quote_cnt INT NOT NULL,
                    created_time INT NOT NULL
                    )''')
        conn.commit()

        cur.execute('''Create table users
                    (id TEXT PRIMARY KEY NOT NULL,
                    tfidf_vec TEXT NOT NULL,
                    w2v_vec TEXT NOT NULL,
                    bert_vec TEXT NOT NULL,
                    verified INT NOT NULL,
                    follower_cnt INT NOT NULL,
                    following_cnt INT NOT NULL,
                    tweet_cnt INT NOT NULL,
                    listed_cnt INT NOT NULL,
                    created_time INT NOT NULL
                    )''')
        conn.commit()
    except Exception as e:
        print(repr(e))


def is_token_valid(token):
    if token in tfidf_stop_words:
        return False
    if len(token) == 1 and token not in 'qwertyuiopasdfghjklzxcvbnm?!':
        return False
    return True


def tweet_normalize_and_tokenize(text):
    norm_text = normalizeTweet(text)
    norm_tokens = [normalizeToken(token).replace('#', '').lower() for token in norm_text.split(' ')]
    norm_tokens = [token for token in norm_tokens if len(token) > 0]
    return norm_tokens


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
        return [0.0] * 300  # vec2str([0.0]*300)

    vectors = [embs[key].reshape(1, -1) if key in embs else np.zeros([1, 300]) for key in tokens]
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


def get_tfidf_embedding(tokens, word_id_dict, topn_idf_dict, n_max=5000):
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

    return vec2str(tfidf_sparse_vec)  # tfidf_vec#tfidf_sparse_vec


def get_sentiment_embedding(text):
    pass


# 对post和user节点进行预处理，然后存到数据库中

def get_existing_set(cur, table_name):
    c = cur.execute('''select id from %s''' % table_name)
    uid_set = set([str(r[0]) for r in c])
    return uid_set


def read_existing_set_from_hdf5(hdf5_path):
    if os.path.exists(hdf5_path):
        f = h5py.File(hdf5_path, 'r')
        existing_set = set(f.keys())
        f.close()
        return existing_set
    else:
        return set()

    # read existing ids


def get_id_set(file_name):
    if not os.path.exists(file_name):
        return set()
    f = open(file_name, 'r')
    id_set = set()
    cnt = 0
    while True:
        if cnt % 100000 == 0:
            print(cnt)
        cnt += 1
        line = f.readline()
        if line:
            fid = int(line.split(',')[0])
            id_set.add(fid)
        else:
            break
    return id_set


def load_json_dataset(data_path, min_n_users=0, min_n_posts=0):
    f = open(data_path)
    # results = json.load(f)
    results = []

    cnt_line = 0
    while True:
        if cnt_line % 3000 == 0:
            print(cnt_line)
        cnt_line += 1
        line = f.readline()

        if line:
            line = json.loads(line)
            if line['n_users'] >= min_n_users and line['n_tweets'] + line['n_retweets'] + line[
                'n_replies'] >= min_n_posts:
                if cnt_line % hvd_size == hvd_rank:
                    results.append(line)
        else:
            break
    f.close()
    return results


def get_id_data(file_path):
    print(file_path)
    f = open(file_path, 'r')
    ids = f.read().split(',')
    ids = [int(i) for i in ids]
    return ids


def get_txt_data(file_path):
    print(file_path)
    f = open(file_path, 'r')
    cnt = 0
    vec_list = []
    while True:
        if cnt % 100000 == 0:
            print(cnt)

        line = f.readline()
        if line:
            cnt += 1
            vals = line.split(',')
            vec = [float(x) for x in vals]
            vec_list.extend(vec)
        else:
            break
    print('num:', cnt)
    return np.array(vec_list).reshape(cnt, -1)


def get_word_embeddings(texts):
    all_vecs = []
    for i in range(len(texts)):
        if i % 10000 == 0:
            mprint('%s/%s' % (i, len(texts)))
        text = texts[i]
        tokens = tweet_normalize_and_tokenize(text)
        word_vec = get_word_embedding(tokens, google_emb, mode='mean')
        all_vecs.append(word_vec)
    all_vecs = np.array(all_vecs, dtype=np.float)
    return all_vecs


def get_bert_embeddings(texts, batch_size):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        if i % (batch_size * 10) == 0:
            mprint('%s/%s' % (i, len(texts)))
        texts_batch = texts[i:i + batch_size]
        vecs_batch = get_bert_embedding_batch(texts_batch, mode='mean')
        all_vecs.extend(vecs_batch)
    all_vecs = np.array(all_vecs, dtype=np.float)
    return all_vecs


def get_tfidf_embeddings(texts, word_id_dict, topn_idf_dict):
    all_vecs = []
    for i in tqdm(range(len(texts))):
        text = texts[i]
        tokens = tweet_normalize_and_tokenize(text)
        word_vec = get_tfidf_embedding(tokens, word_id_dict, topn_idf_dict, n_max=5000)
        all_vecs.append(word_vec)
    all_vecs = '\n'.join(all_vecs)
    return all_vecs


# tfidf=False
# tfidf=False
# w2v=True
# w2v=False
# bert=True
# bert=False

# mode = 'tfidf'
batch_size = 640

hvd_rank = 0
hvd_size = 1
# if mode=='bert':
import horovod.torch as hvd

hvd.init()
hvd_rank = hvd.rank()
hvd_size = hvd.size()
hvd_local_rank = hvd.local_rank()
if torch.cuda.is_available():
    device = torch.device('cuda', hvd_local_rank)
else:
    device = torch.device('cpu')
is_master = (hvd_rank == 0)

overwrite = True
re_tfidf = True
tweet_cnt = 0
existing_tid_set = set()
existing_uid_set = set()
min_n_users = 0
min_n_posts = 0
emb_list = ['meta']  # ['w2v', 'tfidf']



#分别对user description和text进行预处理，分词后，统计高频词，然后对每个词进行编码id
#tweet和retweet直接用一个support token表示
#reply用文本进行表示
#user用文本进行表示   和reply使用不同的文本空间


#delete negative and query word
stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
           "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
           'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
           'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
           'their', 'theirs', 'themselves',
           'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
           'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
           'does', 'did', 'doing', 'a', 'an', 'the', 'and',  'if', 'or',
           'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
           'about', 'against', 'between', 'into', 'through', 'during', 'before',
           'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
           'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
           'here', 'there', 'all', 'any', 'both',
           'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'only', 'own', 'same', 'so', 'than', 'very', 's', 't',
           'can', 'will', 'just', 'now', 'd',
           'll', 'm', 'o', 're', 've', 'y', 'ain',
           '.',',','!',"'s",'...','"',"'",'-',"'re","'m",':','(',')','..',
           '“','”','/']





if __name__ == '__main__':
    print('!!!!!!!!1122!!!!!!!!')
    # a=input('====')
    # conn = sqlite3.connect('./datasets/node_feats_db.db')
    # cur = conn.cursor()
    # if overwrite:
    #    drop_tables(cur)
    # create_tables(cur)

    # conn = sqlite3.connect('./datasets/dataset_db.db')
    # req = 'select * from dataset'
    # results = list(cur.execute(req))
    # cur = conn.cursor()

    # f_p = h5py.File('./datasets/posts.h5','w')
    # f_u = h5py.File('./datasets/users.h5','w')

    # if not overwrite:
    #    existing_tid_set=get_id_set('./datasets/%s_feat_matrix/posts_%s.txt'%(mode,hvd_rank))
    #    existing_uid_set=get_id_set('./datasets/%s_feat_matrix/users_%s.txt'%(mode,hvd_rank))

    #    print('existing tid num',len(existing_tid_set))
    #    print('existing uid num',len(existing_uid_set))

    #    f_p = open('./datasets/%s_feat_matrix/posts_%s.txt'%(mode,hvd_rank),'a+')#'a+')
    #    f_u = open('./datasets/%s_feat_matrix/users_%s.txt'%(mode,hvd_rank),'a+')#'a+')
    # else:

    # columns = 'news_id,title,url,publish_date,source,text,labels,n_tweets,n_retweets,n_replies,n_users,tweets,retweets,' \
    #          'replies,users,data_name'
    # columns = 'news_id,title,url,publish_date,source,text,labels,n_tweets,n_retweets,n_replies,n_users,tweets,retweets,replies,users,retweet_relations,reply_relations,write_relations,tweet_ids,retweet_ids,reply_ids,user_ids,data_name'
    # data_df = load_dataset_and_filter(min_n_users=1,min_n_posts=1,columns=columns,mprint=hprint)

    if overwrite or not os.path.exists('./datasets/text-meta.pkl'):
        data_df = pd.read_csv('./datasets/final_dataset.csv')

        title_df = data_df[['news_id','title']]
        title_df.to_csv('./datasets/titles.csv',header=True)


        num = data_df.shape[0]
        split_size = int(num / hvd_size) + 1
        data_list = data_df.to_dict('records')[hvd_rank * split_size:(hvd_rank + 1) * split_size]
        del (data_df)
        gc.collect()
        barrier(hvd)
        # print(sorted(data_df['data_name_v2'].value_counts().to_dict().items(), key=lambda x: x[0]))
        # data_df=data_df[(data_df['n_tweets']+data_df['n_retweets']+data_df['n_replies'])>=5]
        # print(sorted(data_df['data_name_v2'].value_counts().to_dict().items(), key=lambda x: x[0]))

        # for mode in ['w2v','bert','tfidf']

        print('rank %s, data number: %s' % (hvd_rank, len(data_list)))

        # for tfidf

        # all_size=len(data_list)
        # piece_size=int(all_size/hvd_size)+1
        # data_list=data_list[hvd_rank*piece_size:(hvd_rank+1)*piece_size]

        tqdm_data_list = tqdm(data_list) if hvd_rank == 0 else data_list

        # posts_list=[]
        # users_list=[]
        # post_ids_list=[]
        # user_ids_list=[]
        all_post_ids = []
        all_user_ids = []
        all_post_texts = []
        all_user_texts = []
        all_post_metas = []
        all_user_metas = []
        all_post_types=[]

        all_user_jsons=[]

        for data in tqdm_data_list:
            tweets = json.loads(data['tweets'])
            replies = json.loads(data['replies'])
            retweets = json.loads(data['retweets'])
            users = json.loads(data['users'])

            used_user_ids = set([tw['author_id'] for tw in tweets + retweets + replies])
            user_ids = set([us['id'] for us in users])
            missed_user_ids = list(used_user_ids - user_ids)
            if len(missed_user_ids) > 0:
                print(data['news_id'], len(missed_user_ids), len(tweets), len(retweets), len(replies), len(users))
                missed_users = client.get_users(ids=missed_user_ids, user_fields=user_fields).data
                missed_users = [dict(x) for x in missed_users]
                for us in missed_users:
                    us['created_at'] = us['created_at'].timestamp()
                users = users + missed_users

            # if not overwrite:
            tweets = [tw for tw in tweets if tw['id'] not in existing_tid_set]
            retweets = [tw for tw in retweets if tw['id'] not in existing_tid_set]

            replies = [tw for tw in replies if tw['id'] not in existing_tid_set]
            users = [us for us in users if us['id'] not in existing_uid_set]

            if len(tweets) + len(retweets) + len(replies) == 0:
                continue

            posts = tweets + retweets + replies

            # process posts
            posts_types = [1]*len(tweets)+[2]*len(retweets)+[3]*len(replies)
            posts_ids = [tw['id'] for tw in posts]
            posts_texts = [tw['text'] for tw in posts]
            posts_retweet_counts = [tw['public_metrics']['retweet_count'] for tw in posts]
            posts_reply_counts = [tw['public_metrics']['reply_count'] for tw in posts]
            posts_like_counts = [tw['public_metrics']['like_count'] for tw in posts]
            posts_quote_counts = [tw['public_metrics']['quote_count'] for tw in posts]
            posts_timestamps = [tw['created_at'] for tw in posts]
            # posts_tokens = [tweet_normalize_and_tokenize(text) for text in posts_texts]
            min_time = min(posts_timestamps)
            posts_times = [t - min_time for t in posts_timestamps]

            posts_metas = list(
                zip(posts_retweet_counts, posts_reply_counts, posts_like_counts, posts_quote_counts, posts_times))




            # process users
            users_ids = [us['id'] for us in users]
            users_texts = [us['description'] for us in users]
            description_lens = [len(us['description']) for us in users]
            verified_list = [int(us['verified']) for us in users]
            followers_counts = [us['public_metrics']['followers_count'] for us in users]
            following_counts = [us['public_metrics']['following_count'] for us in users]
            tweet_counts = [us['public_metrics']['tweet_count'] for us in users]
            listed_counts = [us['public_metrics']['listed_count'] for us in users]
            user_created_times = [us['created_at'] - 1288834974657 / 1000 for us in users]
            # users_tokens = [tweet_normalize_and_tokenize(text) for text in users_texts]

            users_metas = list(
                zip(description_lens,verified_list, followers_counts, following_counts, tweet_counts, listed_counts, user_created_times))

            all_texts = posts_texts + users_texts
            # all_tokens = posts_tokens+users_tokens

            for i in range(len(posts_ids)):
                if posts_ids[i] not in existing_tid_set:
                    all_post_ids.append(posts_ids[i])
                    all_post_texts.append(posts_texts[i])
                    all_post_metas.append(posts_metas[i])
                    all_post_types.append(posts_types[i])

            for i in range(len(users_ids)):
                if users_ids[i] not in existing_uid_set:
                    all_user_ids.append(users_ids[i])
                    all_user_texts.append(users_texts[i])
                    all_user_metas.append(users_metas[i])
                    all_user_jsons.append(users[i])

            existing_tid_set = existing_tid_set.union(set(posts_ids))
            existing_uid_set = existing_uid_set.union(set(users_ids))

        barrier(hvd)
        print(hvd_rank, 'post num', len(all_post_ids))
        print(hvd_rank, 'user num', len(all_user_ids))

        all_list = [all_post_texts,all_post_types,all_post_ids,all_post_metas,all_user_texts,all_user_ids,all_user_metas]

        def gather_list(data_list):
            all_list = hvd.allgather_object(data_list)
            all_list = sum(all_list, [])
            return all_list

        for i in range(len(all_list)):
            all_list[i]=gather_list(all_list[i])
            #去重

        all_post_texts, all_post_types, all_post_ids, all_post_metas, all_user_texts, all_user_ids, all_user_metas=all_list
        print('before dropdup, %s,%s'%(len(all_post_ids),len(all_user_ids)))

        def drop_dup(ids,texts,metas,types=None):
            id_set=set()
            ids_d=[]
            texts_d = []
            metas_d=[]
            types_d=[]
            for i in range(len(ids)):
                if ids[i] not in id_set:
                    ids_d.append(ids[i])
                    texts_d.append(texts[i])
                    metas_d.append(metas[i])
                    if types:
                        types_d.append(types[i])
                id_set.add(ids[i])
            return ids_d,texts_d,metas_d,types_d


        all_post_ids, all_post_texts, all_post_metas, all_post_types=drop_dup(all_post_ids, all_post_texts, all_post_metas, all_post_types)
        all_user_ids, all_user_texts, all_user_metas,_ = drop_dup(all_user_ids, all_user_texts, all_user_metas)
        print('after dropdup, %s,%s'%(len(all_post_ids),len(all_user_ids)))

        all_list = [all_post_texts,all_post_types,all_post_ids,all_post_metas,all_user_texts,all_user_ids,all_user_metas]


        if hvd_rank==0:

            f=open('./datasets/text-meta.pkl','wb')
            pickle.dump(all_list,f)
            f.close()

            dump_pkl(all_user_jsons,'./datasets/all_users.pkl')

        barrier(hvd)

    else:
        print('load pkl')
        f = open('./datasets/text-meta.pkl', 'rb')
        all_post_texts,all_post_types,all_post_ids,all_post_metas,all_user_texts,all_user_ids,all_user_metas=pickle.load(f)
        f.close()


        #f_p = h5py.File('./datasets/%s_feat_matrix/posts_%s.h5' % (mode, hvd_rank), 'w')  # 'a+')
        #f_u = h5py.File('./datasets/%s_feat_matrix/users_%s.h5' % (mode, hvd_rank), 'w')  # 'a+')

        barrier(hvd)
        #all_post_vecs = get_word_embeddings(all_post_texts)
        #all_user_vecs = get_word_embeddings(all_user_texts)


    if is_master:
        print('writing user_list')
        f = open('user_list.txt', 'w')
        print(len(all_user_ids))
        all_uid_str = '\n'.join([str(uid) for uid in all_user_ids])
        f.write(all_uid_str)
        f.close()


    if False:
        #只使用reply的text
        mprint(len(all_post_texts))
        #不是reply全部替换成空字符串
        all_post_texts=[text if all_post_types[i]==3 else '' for i,text in enumerate(all_post_texts)]
        mprint(len(all_post_texts))


        def texts2ids(all_texts,all_types=None,min_freq=5):#tweet和user得不一样。。。

            all_tokens=[]
            token_dict={}
            for i in tqdm(range(len(all_texts))):
                #if i % 10000 == 0:
                #    mprint('%s/%s' % (i))
                text = all_texts[i]

                if text == '':
                    tokens=[]
                else:
                    tokens = tweet_normalize_and_tokenize(text)[:130]
                all_tokens.append(tokens)


                for tk in tokens:
                    if tk not in token_dict:
                        token_dict[tk]=1
                    else:
                        token_dict[tk]+=1

            token_nums = [len(tokens) for tokens in all_tokens]
            #long_tokens = [tokens for tokens in all_tokens if len(tokens)>100]
            token_freq_list = sorted(token_dict.items(),key=lambda x:x[1],reverse=True)
            token_freq_list = [x for x in token_freq_list if x[0] not in stopwords]


            mprint(len(token_freq_list))
            filtered_token_freq_list = [x for x in token_freq_list if x[1]>min_freq]
            mprint(len(filtered_token_freq_list))

            if all_types:
                token_id_dict={'[#PAD#]':0,'[#TWEET#]':1,'[#RETWEET#]':2,'[#REPLY#]':3,'[#OOV#]':4}
            else:
                token_id_dict={'[#PAD#]':0,'[#USER#]':1,'[#OOV#]':2}
            for tk in filtered_token_freq_list:
                token_id_dict[tk[0]]=len(token_id_dict)
            #空字符串用0000表示

            def map_token2id(tokens, text_type):
                ids = [token_id_dict[tk] if tk in token_id_dict else token_id_dict['[#OOV#]'] for tk in tokens]

                ids = ids if len(ids)>0 else [text_type]
                return ids

            if all_types:
                all_ids = [map_token2id(tokens, all_types[i])[:100] for i,tokens in tqdm(enumerate(all_tokens))]
            else:
                all_ids = [map_token2id(tokens, 0)[:100] for i,tokens in tqdm(enumerate(all_tokens))]



            return all_ids,all_tokens,filtered_token_freq_list,token_id_dict


        all_post_token_ids,all_post_tokens,filtered_post_token_freq_list,post_token_id_dict = texts2ids(all_post_texts,all_post_types)
        all_user_token_ids,all_user_tokens,filtered_user_token_freq_list,user_token_id_dict = texts2ids(all_user_texts,min_freq=10)

        all_data = [all_post_ids, all_post_token_ids, all_post_metas, all_post_sentiment, all_user_ids,
                    all_user_token_ids, all_user_metas]
        barrier(hvd)

        if is_master:
            f = open('./datasets/context_features.pkl', 'wb')
            pickle.dump(all_data, f)
            f.close()








    #group


