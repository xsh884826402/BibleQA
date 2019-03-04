#shell 脚本
import os
if __name__ == "__main__":
    word2vec_model_list = ['word2vec_model_1','word2vec_model_0.75','word2vec_model_0.5','word2vec_model_0.25']
    score_dict_list = ['score_dict_1', 'score_dict_0.75','score_dict_0.5','score_dict_0.25']
    score_weight_dict_list = ['score_weight_dict_1','score_weight_dict_0.75','score_weight_dict_0.5','score_weight_dict_0.25']
    text_list = ['text.pkl','text_0.75.pkl','text_0.5.pkl','text_0.25.pkl']
    tf_idf_list = ['tf_idf.pkl', 'tf_idf_0.75.pkl','tf_idf_0.5.pkl','tf_idf_0.25.pkl']
    for i in range(4):
        os.system('python Calculate_Quality.py'+ ' '+ word2vec_model_list[i]+ ' ' + score_dict_list[i]+ ' ' +score_weight_dict_list[i])
    # os.system('python Calculate_Quality.py word2vec_model_1 score_dict_1'+' score_weight_dict_1')
    # for i in range(4):
    #     # print('python Tf_IDF.py'+ ' ' +text_list[i] + ' '+tf_idf_list[i])
    #     os.system('python Tf_IDF.py' + ' ' + text_list[i] + ' ' + tf_idf_list[i])