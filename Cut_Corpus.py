#   对得到的语料进行切分，得到不同大小的语料
#   cut_corpus 输入文本列表和截断长度
#               使用random.sample 对文本列表进行随机采样
#               输出长度缩小的文本列表,将新的文本列表存储到all_text_dir 指定的路径里
# sys.argv[1]是存储的路径，sys.argv[2]是原长度的k倍，k为0~1的小数
import pickle
import random
import sys
def cut_corpus(text,fraction):
    text_length = fraction*len(text)
    new_text = random.sample(text, int(text_length))
    return new_text


if __name__ == "__main__":
    all_text_dir_0 = './data/word_vector/text.pkl'
    all_text_dir = './data/word_vector/'
    all_text_dir = all_text_dir + sys.argv[1]
    with open(all_text_dir_0, 'rb') as f:
        all_text = pickle.load(f)
    print('Before cut',len(all_text))
    new_text = cut_corpus(all_text, float(sys.argv[2]))
    print('指定路径',all_text_dir)
    with open(all_text_dir, 'wb') as f:
        pickle.dump(new_text, f)
    print("After Cut", len(new_text))