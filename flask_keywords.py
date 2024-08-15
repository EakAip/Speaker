# 接口：8003

# 返回停用词和语气词

# 所有词汇转为小写

# 初始化 TF-IDF 向量化器，禁用自动转换为小写


from flask import Flask, request, jsonify
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

app = Flask(__name__)

# 加载停用词
def load_stopwords(path):
    stopwords = set()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords

# 关键词提取函数
def extract_keywords(text):
    # 读取停用词
    stopwords = load_stopwords('./configs/stop_words.txt')
    
    # 使用 jieba 进行分词
    words = list(jieba.cut(text))
    # 所有词汇转为小写
    words = [word.lower() for word in words]
    
    # 过滤出停用词和非停用词
    stopwords_in_text = [word for word in words if word in stopwords]  # 停用词
    non_stopwords = [word for word in words if word not in stopwords]  # 关键词
    text_cut = " ".join(non_stopwords)
    
    # 初始化 TF-IDF 向量化器 ，禁用自动转换为小写
    vectorizer = TfidfVectorizer(lowercase=False)
    
    # 计算 TF-IDF
    tfidf_matrix = vectorizer.fit_transform([text_cut])  # 计算关键词TF-IDF
    
    # 获取每个词的 TF-IDF 值
    words_tfidf = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
    
    # 对关键词按照 TF-IDF 值排序
    sorted_words_tfidf = sorted(words_tfidf.items(), key=lambda item: item[1], reverse=True)
    
    # 获取原始文本中每个词的出现次数（包括停用词和非停用词）
    word_counts = Counter(words)
    
    # 获取前 50 个关键词及其出现次数
    top_keywords = {word: word_counts[word] for word, _ in sorted_words_tfidf[:60]}
    
    # 获取停用词及其出现次数
    stopwords_counts = {word: word_counts[word] for word in stopwords_in_text if word_counts[word] > 0}
    
    return top_keywords, stopwords_counts

@app.route('/keywords', methods=['POST'])
def keywords():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"code": 5, "msg": "No text provided", "data": {}}), 400
    
    try:
        keywords, stopwords = extract_keywords(text)
        keywords_str = ",".join([f"{word}:{count}" for word, count in keywords.items()])
        stopwords_str = ",".join([f"{word}:{count}" for word, count in stopwords.items()])
        return jsonify({"code": 0, "msg": "OK", "data": {"keywords": keywords_str, "stopwords": stopwords_str}})
    except Exception as e:
        return jsonify({"code": 5, "msg": str(e), "data": {}}), 500

if __name__ == '__main__':
    app.run(port=8003, host='0.0.0.0')

