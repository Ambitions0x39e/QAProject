import token
from urllib.parse import _ResultMixinStr
import pandas as pd
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
import jieba
import jieba.analyse
from data_models import QueryResult
import numpy as np

stop_words = []
csv_path = 'func_desc.csv'
sentences, tokenized_sentences, dictionary, tfidf, index, word2vec_model = None, None, None, None, None, None

'''
TODO:
- 真实数据
- 
'''

def build_word2vec_model(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
    return model

def expand_query_with_synonyms(query, word2vec_model):
    query_words = jieba.lcut(query)
    expanded_words = []
    for word in query_words:
        try:
            # 获取相似词
            similar_words = word2vec_model.wv.most_similar(word, topn=3)
            expanded_words.extend([w for w, _ in similar_words])
        except:
            continue
    return query_words + expanded_words

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.iloc[:, 0].tolist()

def preprocess_text(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        words = jieba.lcut(str(sentence))  # 确保输入是字符串
        words = [word for word in words if word not in stop_words]
        tokenized_sentences.append(words)
    return tokenized_sentences

def build_model(tokenized_sentences):
    dictionary = corpora.Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(text) for text in tokenized_sentences]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    index = similarities.MatrixSimilarity(corpus_tfidf)
    return dictionary, tfidf, index

def process_query(query, dictionary, tfidf):
    # 提取关键词并保持原有权重
    keywords = jieba.analyse.extract_tags(query, topK=5, withWeight=True)
    query_words = []
    
    # 增加关键词权重影响
    for word, weight in keywords:
        # 根据权重值动态调整重复次数
        repeat_times = int(weight * 15)  # 增加权重倍数
        query_words.extend([word] * repeat_times)
    
    # 添加原始分词结果，但赋予较低权重
    original_words = jieba.lcut(query)
    query_words.extend(original_words)
    
    # 转换为bow格式并计算tfidf
    query_bow = dictionary.doc2bow(query_words)
    query_tfidf = tfidf[query_bow]
    return query_tfidf

def similarity_TF_IDF(query_tfidf, index, sentences, query, top_n=3):
    sims = index[query_tfidf]
    
    # 增加关键词匹配权重
    keywords = jieba.analyse.extract_tags(query, topK=3)  # 提取查询中的关键词
    keyword_weights = []
    
    for s in sentences:
        weight = 0
        for keyword in keywords:
            if keyword in str(s):
                weight += 0.3  # 每个关键词命中增加权重
        keyword_weights.append(weight)
    
    # 长度惩罚因子，降低权重影响
    length_penalties = [1 / (1 + abs(len(str(s)) - len(str(query)))) for s in sentences]
    
    # 调整权重分配
    weight_similarity = 0.8    # 提高相似度权重
    weight_keyword = 0.15      # 添加关键词权重
    weight_length = 0.05       # 降低长度影响
    
    # 组合不同特征的得分
    combined_scores = [
        (sim * weight_similarity + 
         kw * weight_keyword + 
         lp * weight_length)
        for sim, kw, lp in zip(sims, keyword_weights, length_penalties)
    ]
    
    sim_scores = list(enumerate(combined_scores))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in sim_scores[:top_n]:
        results.append({
            'sentence': sentences[idx],
            'score': round(float(score), 3)
        })
    return results

def jaccard_similarity_gensim(a_tokens, b_tokens):
        set_a = set(a_tokens)
        set_b = set(b_tokens)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)
      
def similarity_Jaccard(sentences, query, tokenized_sentences, top_n=3):
    # 确保query为字符串并去除None等异常情况
    if not isinstance(query, str):
        query = str(query) if query is not None else ""
    try:
        query = query.encode('utf-8').decode('utf-8')
    except Exception:
        query = str(query)
    # 分词并去停用词
    query_tokens = jieba.lcut(query)
    query_tokens = [word for word in query_tokens if word not in stop_words]

    sims = []
    for idx, sent_tokens in enumerate(tokenized_sentences):
        sim = jaccard_similarity_gensim(query_tokens, sent_tokens)
        sims.append((idx, sim))

    top_indices = sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for i, score in top_indices:
        results.append({
            'sentence': sentences[i],
            'score': round(float(score), 3)
        })
    return results

def init():
    global stop_words, sentences, tokenized_sentences, dictionary, tfidf, index, word2vec_model
    with open("stopword.txt", 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f.readlines()]
    sentences = load_data(csv_path)
    tokenized_sentences = preprocess_text(sentences)
    dictionary, tfidf, index = build_model(tokenized_sentences)
    word2vec_model = build_word2vec_model(tokenized_sentences)

def query_tfidf(query_string, ans_amount=3):
    global stop_words, sentences, tokenized_sentences, dictionary, tfidf, index, word2vec_model
    if dictionary is None or tfidf is None or index is None or word2vec_model is None:
        init()
    # 扩展查询，但限制扩展词的影响
    expanded_words = expand_query_with_synonyms(query_string, word2vec_model)
    # 原始查询占更大权重
    expanded_query = query_string + ' ' + ' '.join(expanded_words)
    
    
    # Process query via tf-idf algorithm
    query_tfidf = process_query(expanded_query, dictionary, tfidf)
    results = similarity_TF_IDF(query_tfidf, index, sentences, query_string, top_n=ans_amount)
    
    lst=[]
    
    if results:
        print("\n找到的前三个最相似的功能描述：") 
        # Method TF-IDF
        for i, result in enumerate(results, 1):
            lst.append(QueryResult(result['sentence'], result['score']))

    return lst

def query_jaccard(query_string, ans_amount=3):
    global stop_words, sentences, tokenized_sentences, dictionary, tfidf, index, word2vec_model
    if dictionary is None or tfidf is None or index is None or word2vec_model is None:
        init()
    # 扩展查询，但限制扩展词的影响
    expanded_words = expand_query_with_synonyms(query_string, word2vec_model)
    # 原始查询占更大权重
    expanded_query = query_string + ' ' + ' '.join(expanded_words)
    
    
    results = similarity_Jaccard(sentences, query_string, tokenized_sentences, top_n=ans_amount)
    
    
    lst=[]
    
    if results:
        print("\n找到的前三个最相似的功能描述：") 
        for i, result in enumerate(results, 1):
            lst.append(QueryResult(result['sentence'], result['score']))
    
    return lst

def main():
    init()
    
    while True:
        question = input("请输入您的问题（输入'q'退出）：")
        if question.lower() == 'q':
            break
            
        query_results = query(question,3)
        
        for _ in query_results:
            print(_.debug_repr())
            # print(repr(_))      

if __name__ == "__main__":
    main()