import csv
import re
import nltk

# # 初次使用需下载以下资源
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

"""
# 数据预处理类，包含实验过程中需要的数据预处理函数，按需调用和补充
"""

class DataPreprocessor:
    def __init__(self, path_of_file, max_samples = None):
        self.file_path = path_of_file
        self.texts = []
        self.labels = []
        self.max_samples = max_samples  # 最多加载的文本数，None表示不限制
        self.processed_texts = []
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self._load_datas()

    # public
    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    def get_dataset(self):
        """返回 (text, label) 的 zip 数据"""
        return list(zip(self.texts, self.labels))

    def get_processed_texts(self):
        return self.processed_texts

    def get_processed_dataset(self):
        """预处理后的 tokens + 标签"""
        return list(zip(self.processed_texts, self.labels))

    # private
    def _load_datas(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                text = row['post_text'].strip()
                label_str = row['label'].strip().lower()

                if label_str == 'fake':
                    label = 0
                elif label_str == 'real':
                    label = 1
                else:
                    print(f"警告：未知标签 {label_str}，跳过该行")
                    continue

                self.texts.append(text)
                self.labels.append(label)

                # 如果设置了最大条数限制并达到了数量，提前退出
                if self.max_samples is not None and len(self.texts) >= self.max_samples:
                    break

            # 输出真假新闻数据
            fake_count = self.labels.count(0)
            real_count = self.labels.count(1)
            print(f"====== 加载完成：共 {len(self.labels)} 条新闻 ======")
            print(f" 假新闻数量: {fake_count}")
            print(f" 真新闻数量: {real_count}")
            print("  数据示例：", self.texts[0], "标签：", self.labels[0])

            self.processed_texts = [self._preprocess_text(t) for t in self.texts]

    def _preprocess_text(self, text):
        # 清洗非字母字符，转小写
        text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
        tokens = text.split()

        # 去停用词 + 去短词
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        # 词形还原
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens