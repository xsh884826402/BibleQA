# 获取圣经的语料，然后训练成词向量
#   输入：圣经的语料
#   function：将得到的圣经语料，使用gensim中的Word2Vec去训练语料，得到词向量
#   输出：将得到的词向量存储下来，路径由word2vec_dir指定
#   sys.argv[1] 词向量的存储路径， sys.argv[2] 语料的路径
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
# from gensim.test.test_utils import get_tmpfile
import re
import json
import pickle
import sys
import locale
def read_data():


    #in list formats
    questions = []
    answers = []
    labels = []

    with open('data/bible_qa/bible_qa_list_3.json', 'r') as f:
        data = json.load(f)
        for item in data:
            for version_answer in item["answers"]:

                questions.append(item["question"])
                answers.append(version_answer)
                labels.append(item["labels"])


    return {"Q": questions, "A": answers, "L": labels}
def get_all_texts(data):

    # finally, vectorize the text samples into a 2D integer tensor
    all_questions = data["Q"]
    all_answers = data["A"]
    labels = data["L"]

    all_texts = []

    for question_list in all_questions:
        for question in question_list:
            all_texts = all_texts + [clean_sentence(question)]
            break

    for answer_list in all_answers:
        for answer in answer_list:
            all_texts = all_texts + [clean_sentence(answer)]
    return all_texts
def clean_sentence(text):
    #print("before: ", text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    # changing contractions
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"\'s", "", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)

    text = re.sub(' +', ' ', text)
    text = text.split()

    text = [w.lower() for w in text]

    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]

    text = " ".join(text)

    #print("after: ", text)

    return text
if __name__ == "__main__":
    word2vec_dir = './data/word_vector/'
    all_text_dir = './data/word_vector/'
    word2vec_dir = word2vec_dir + sys.argv[1]
    all_text_dir = all_text_dir + sys.argv[2]
    # data = read_data()
    # all_text = get_all_texts(data)
    # with open(all_text_dir, 'wb') as f:
    #     pickle.dump(all_text,f)

    with open(all_text_dir, 'rb') as f:
        all_text = pickle.load(f)
    all_text_list = []
    for item in all_text:
        temp = item.split()
        all_text_list.append(temp)

    #Debug
    model = Word2Vec(all_text_list,size=128,window=5,min_count=2)
    model.wv.save(word2vec_dir)
    model_new = KeyedVectors.load(word2vec_dir)
    bible_vocab = model_new.vocab
    print('Length', len(bible_vocab),len(set(bible_vocab)))














