#z评价词向量的好坏
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
import pickle
embeddir = 'data/embeddings/'
bible_model = KeyedVectors.load(embeddir+'bible_word2vec')
bible_vocab = bible_model.vocab
score_dict = {}
score_dict_dir = 'Score_dict'
#词向量的最相似的n个词
most_similar_num = 4
#计算语义距离的函数，可以按需要修改。
def calculate_ave(distance, n):
    return distance/n
def calculate_ave_weight(n):
    return n
for word in bible_vocab:
    word_wn0 = wn.synsets(word)
    #有的词在wordnet中没有语义
    if word_wn0 == []:
        continue
    print(word_wn0)
    # 取语义网络中第一个语义
    word_wn = word_wn0[0]

    word_list = bible_model.most_similar(word)
    #词向量中最相似的n个词
    word_list = word_list[0:most_similar_num]
    print('word_list', word_list)
    syn_list = []
    count = 0
    distance = float(0)
    #sum_n 用来计算加权的语义距离
    sum_n = 0
    distance_weight = 0
    for i in range(len(word_list)):
        #w代表词，n代表余弦距离
        w , n = word_list[i]
        sum_n += n
        # print(w, n, i)
        if (wn.synsets(w) == []):
            continue
        else:
            count += 1
        w_wn = wn.synsets(w)[0]
        #计算距离的方式有很多种，暂时取平均值
        temp = word_wn.path_similarity(w_wn)
        #有的两个词 是无法计算语义距离的，所以需要做一下判断
        if temp is None:
            distance += 0
        else:
            distance +=temp
            distance_weight += n*temp
        print('distance', distance_weight,)
    if count ==0:
        continue
    ave_distance = calculate_ave(distance, count)
    ave_distance_weight = calculate_ave(distance_weight,sum_n)
    score_dict[word] = ave_distance
    print('word:',word, 'ave_distance:', ave_distance)
    print('word',word, 'ave_distance_weight', ave_distance_weight)


    for item in wn.synsets(word):
        syn_list.extend(item.lemma_names())
    print(set(syn_list))
    syn = wn.synsets(word)
    print(syn)

