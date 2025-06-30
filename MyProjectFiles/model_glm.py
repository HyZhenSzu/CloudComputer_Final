from transformers import AutoTokenizer, AutoModel
from data_preprocess import DataPreprocessor
from tqdm import tqdm


# 两个数据集的posts文件路径
DEVSET_POST_FILE_PATH = "E:/HyZhen/pythonProject/CloudProject/datas/twitter_dataset/devset/posts.txt"
TESTSET_POST_FILE_PATH = "E:/HyZhen/pythonProject/CloudProject/datas/twitter_dataset/testset/posts_groundtruth.txt"


class ChatGLMWrapper:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda().eval()

    def chat(self, prompt, max_new_tokens=32):
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            max_new_tokens=max_new_tokens
        )
        return response.strip().lower()


class NewsClassifier:
    def __init__(self, glm: ChatGLMWrapper):
        self.glm = glm

    def predict(self, text: str) -> int:
        prompt = (
            "请你根据常识判断以下新闻是否可信。\n"
            "你只能回答“真实”或“虚假”，不要给出任何解释。\n\n"
            f"新闻内容：{text}\n\n"
            "请直接回答：真实 或 虚假"
        )
        response = self.glm.chat(prompt)

        if "虚假" in response:
            return 0
        elif "真实" in response:
            return 1
        else:
            print("⚠️ 无法判断输出：", response)
            return -1

class NewsClassifierWithSentiment:
    def __init__(self, glm: ChatGLMWrapper):
        self.glm = glm

    def classify_sentiment(self, text: str) -> str:
        prompt = (
            "请分析以下这条新闻的情感倾向。"
            "你只能回答：正面、中立 或 负面，不要给出任何解释。\n\n"
            f"{text}\n\n"
            "请直接回答情感类别："
        )
        response = self.glm.chat(prompt)
        if "负面" in response:
            return "负面"
        elif "正面" in response:
            return "正面"
        elif "中立" in response:
            return "中立"
        else:
            print("⚠️ 无法识别情感输出：", response)
            return "未知"

    def predict(self, text: str):
        sentiment = self.classify_sentiment(text)
        prompt = (
            f"这条新闻的情感倾向是：{sentiment}。\n"
            "请结合其内容和情感倾向判断该新闻是否可信，只回答“真实”或“虚假”，不要给出任何解释。\n\n"
            f"新闻内容：{text}\n\n"
            "请直接回答：真实 或 虚假"
        )
        response = self.glm.chat(prompt)

        if "虚假" in response:
            return 0,sentiment
        elif "真实" in response:
            return 1,sentiment
        else:
            print("⚠️ 无法判断输出：", response)
            return -1,sentiment

if __name__ == '__main__':
    # 加载模型
    glm = ChatGLMWrapper()

    # 加载数据
    processor = DataPreprocessor(DEVSET_POST_FILE_PATH)
    texts = processor.get_texts()
    labels = processor.get_labels()

    # 选择分类器
    # classifier = NewsClassifier(glm)
    classifier = NewsClassifierWithSentiment(glm)

    # 准确率评估
    correct = 0
    correct_fake = 0
    correct_real = 0
    total = len(texts)
    total_fake = labels.count(0)
    total_real = labels.count(1)

    # for i, (text, label) in enumerate(zip(texts, labels)):
    #     pred = classifier.predict(text)
    #     if pred == label:
    #         correct += 1
    #         if label == 0:
    #             correct_fake += 1
    #         else:
    #             correct_real += 1
    #     print(f"[{i + 1}/{total}] GT: {label} | Pred: {pred}")

    for i, (text, label) in enumerate(zip(texts, labels)):
        pred, sentiment = classifier.predict(text)
        if pred == label:
            correct += 1
            if label == 0:
                correct_fake += 1
            else:
                correct_real += 1
        print(f"[{i + 1}/{total}] GT: {label} | Pred: {pred} | Sentiment: {sentiment}")

    # 输出结果
    accuracy = correct / total
    accuracy_fake = correct_fake / total_fake if total_fake else 0
    accuracy_real = correct_real / total_real if total_real else 0

    print("\n==== 结果统计 ====")
    print(f"总样本数: {total}")
    print(f"总准确率: {accuracy:.4f}")
    print(f"假新闻准确率: {accuracy_fake:.4f}")
    print(f"真新闻准确率: {accuracy_real:.4f}")
