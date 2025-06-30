from gensim import corpora, models
from data_preprocess import DataPreprocessor
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModel

# 两个数据集的posts文件路径
DEVSET_POST_FILE_PATH = "E:/HyZhen/pythonProject/CloudProject/datas/twitter_dataset/devset/posts.txt"
TESTSET_POST_FILE_PATH = "E:/HyZhen/pythonProject/CloudProject/datas/twitter_dataset/testset/posts_groundtruth.txt"

class LDAModeler:
    def __init__(self, tokenized_texts, num_topics=3):
        self.tokenized_texts = tokenized_texts
        self.num_topics = num_topics
        self.dictionary = None
        self.corpus = None
        self.model = None
        self._build_model()

    def _build_model(self):
        self.dictionary = corpora.Dictionary(self.tokenized_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokenized_texts]

        self.model = models.LdaModel(
            self.corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=10,
            random_state=42
        )

    def print_topics(self, num_words=8):
        print("==== 模型主题分布 ====")
        topics = self.model.print_topics(num_words=num_words)
        for idx, topic in topics:
            print(f"主题 {idx + 1}: {topic}")

    def get_document_topics(self):
        """返回每个文档对应的主题分布"""
        return [self.model.get_document_topics(bow) for bow in self.corpus]

    def visualize_pyldavis(self, save_path: str = "lda_vis.html"):
        print(f"====== 正在生成 pyLDAvis 可视化图（保存至 {save_path}）======")
        vis_data = gensimvis.prepare(self.model, self.corpus, self.dictionary)
        pyLDAvis.save_html(vis_data, save_path)

        with open(save_path, 'r', encoding='utf-8') as f:
            html = f.read()
        html = html.replace('width: 1000px;', 'width: 1400px;')  # 放大宽度
        html = html.replace('height: 830px;', 'height: 1000px;')  # 放大高度
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print("可视化完成！")

    def visualize_wordclouds(self):
        for topic_id in range(self.num_topics):
            words_probs = self.model.show_topic(topic_id, topn=20)
            wordcloud = WordCloud(
                background_color='white',
                width=800,
                height=400
            ).generate_from_frequencies(dict(words_probs))

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f"WordCloud - Topic {topic_id + 1}")
            plt.show()

class TopicInterpreter:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        print("加载 ChatGLM3 模型中...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda().eval()

    def explain_topic(self, topic_keywords, topn=8):
        """
        传入一组关键词（按重要性排序），生成该主题的自然语言解释。
        :param topic_keywords: List[str]，例如 ["bomber", "boston", "suspect"]
        :return: str
        """
        keywords = ", ".join(topic_keywords[:topn])
        prompt = (
            f"以下是一组关键词：{keywords}。请根据这些关键词分析它们可能代表的主题，"
            f"并用一句简洁自然的语言描述该主题内容（比如：'该主题主要描述...'），不要回答与问题无关的内容。"
            f"除了人名，全部使用中文进行回答。"
        )

        response, _ = self.model.chat(self.tokenizer, prompt, history=[], max_new_tokens=256)
        return response

    def batch_explain(self, topic_word_lists, topn=8):
        """
        对多个主题进行解释
        :param topic_word_lists: List[List[str]]，每个主题的关键词列表
        :return: List[str]，对应每个主题的解释
        """
        explanations = []
        for idx, keywords in enumerate(topic_word_lists):
            print(f"\n=== 正在解释主题 {idx+1} ===")
            explanation = self.explain_topic(keywords, topn=topn)
            print(f"主题 {idx+1} 解释：{explanation}")
            explanations.append(explanation)
        return explanations

if __name__ == '__main__':
    # 加载数据
    processor = DataPreprocessor(DEVSET_POST_FILE_PATH, max_samples=10)
    tokenized_texts = processor.get_processed_texts()

    # 训练 LDA 模型（设定主题数为 3）
    lda = LDAModeler(tokenized_texts, num_topics=3)
    lda.print_topics()

    # # 生成 pyLDAvis 可视化文件
    # save_path = "E:/HyZhen/pythonProject/CloudProject/MyProjectFiles/VisualizeOutputs/lda_vis.html"
    # lda.visualize_pyldavis(save_path)
    #
    # # 显示每个主题的词云图
    # lda.visualize_wordclouds()

    # 提取每个主题的 top-n 关键词（用于解释）
    topic_word_lists = []
    for topic_id in range(lda.num_topics):
        words_probs = lda.model.show_topic(topic_id, topn=8)
        keywords = [word for word, _ in words_probs]
        topic_word_lists.append(keywords)

    # 使用 ChatGLM3 解释每个主题
    interpreter = TopicInterpreter()
    print("\n====== ChatGLM3 主题解释结果 ======")
    explanations = interpreter.batch_explain(topic_word_lists)

