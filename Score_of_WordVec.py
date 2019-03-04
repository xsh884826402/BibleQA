#根据给定的tf_idf词典（key是单词）和语义网络分数词典（key是单词），计算出该种词向量的分数
#   输入：tf_idf词典，sys.argv[1],语义网络分数词典(sys.argv[2])
#   处理：对于每个单词，用tf_idf,对词向量进行加权
#   输出：每种词向量的分数
import pickle
import sys
from gensim.models import KeyedVectors
tf_idf_basedir = './data/word_vector/'
word2vec_model_basedir = './data/score_dict/'
# tf_idf_dir = tf_idf_basedir + sys.argv[1]
# word2vec_model_dir = word2vec_model_basedir + sys.argv[2]
def function(tf_idf_dir, word2vec_model_dir):
    sum_tf_idf = 0
    sum_score = 0
    sum_score_1 = 0
    with open(tf_idf_dir, 'rb') as f:
        tf_idf_dict = pickle.load(f)
    # word2vec_model_dict = KeyedVectors.load(word2vec_model_dir)
    with open(word2vec_model_dir, 'rb') as f:
        word2vec_model_dict = pickle.load(f)
    flag = 0
    for key in word2vec_model_dict.keys():
        if (key in tf_idf_dict.keys()):
            sum_tf_idf += tf_idf_dict[key]
            sum_score += word2vec_model_dict[key]*tf_idf_dict[key]
        sum_score_1 += word2vec_model_dict[key]
    sum_score = sum_score/sum_tf_idf
    sum_score1 = sum_score_1/len(word2vec_model_dict.keys())

    print("length of tf_idf",len(tf_idf_dict.keys()), "length of word2vec", len(word2vec_model_dict.keys()))
    return sum_score,sum_score_1
if __name__ == "__main__":

    score_dict_list = ['score_dict_1', 'score_dict_0.75', 'score_dict_0.5', 'score_dict_0.25']
    score_weight_dict_list = ['score_weight_dict_1', 'score_weight_dict_0.75', 'score_weight_dict_0.5',
                              'score_weight_dict_0.25']
    tf_idf_list = ['tf_idf,pkl', 'tf_idf_0.75.pkl', 'tf_idf_0.5.pkl', 'tf_idf_0.25.pkl']
    for i in range(4):
        print("第i轮次", i)
        tf_idf_dir = tf_idf_basedir + tf_idf_list[i]
        word2vec_model_dir = word2vec_model_basedir + score_dict_list[i]
        score,score_1 = function(tf_idf_dir, word2vec_model_dir)
        print("Score",score,score_1)