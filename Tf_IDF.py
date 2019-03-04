#   利用sklearn库提取语料的tfidf特征
#   sys.argv[1]读取的text_dir， sys.argv[2] tf_idf词典的存储路径
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import numpy as np
import sys
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
base_dir = './data/word_vector/'
text_dir = base_dir + sys.argv[1]
tf_idf_dir = base_dir + sys.argv[2]
# text_dir = base_dir + sys.argv[1]

with open(text_dir, 'rb',) as f:
    text = pickle.load(f)
tf_idf_dict = transformer.fit_transform(vectorizer.fit_transform(text))
word = vectorizer.get_feature_names()

# print(type(word),word)
# print('TF_IDf', tf_idf_dict)
weight = tf_idf_dict.toarray()
weight_final = np.sum(weight,axis=0)
# print('After sum', np.shape(weight_final),weight_final[0:10])
tf_idf = {}
for i,w in enumerate(word):
    tf_idf[w] = weight_final[i]
with open(tf_idf_dir, 'wb') as f:
    pickle.dump(tf_idf, f)
print("---------------------Success---------------------------------")


