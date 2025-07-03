# run_pipeline.py

from data_preprocess import DataPreprocessor
from image_caption import ImageCaptionGenerator
from model_prompting_tree import TreePromptPredictor

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def main():
    # === 路径配置 ===
    POSTS_PATH = "E:/HyZhen/pythonProject/CloudProject/datas/twitter_dataset/devset/posts.txt"
    IMAGE_FOLDER = "E:/HyZhen/pythonProject/CloudProject/datas/twitter_dataset/devset/images/"

    # === 初始化 ChatGLM ===
    print("\n[1] 初始化 ChatGLM 模型...")
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda().eval()
    predictor = TreePromptPredictor(glm_model=model, tokenizer=tokenizer)

    # === 加载数据 ===
    print("\n[2] 加载文本数据...")
    preprocessor = DataPreprocessor(POSTS_PATH)  # 可调节 max_samples
    texts = preprocessor.get_texts()
    labels = preprocessor.get_labels()
    image_id_lists = preprocessor.get_image_ids()

    # === 图像描述生成 ===
    print("\n[3] 生成图像描述...")
    captioner = ImageCaptionGenerator(image_root=IMAGE_FOLDER)
    image_captions = captioner.caption_images_for_posts(image_id_lists)

    # === 预测并评估 ===
    print("\n[4] 开始预测...")
    correct = 0
    correct_real = 0
    correct_fake = 0
    total_real = labels.count(1)
    total_fake = labels.count(0)

    for i in tqdm(range(len(texts))):
        text = texts[i]
        label = labels[i]
        caption = image_captions[i] or "无图像信息"

        pred_label, sub_conclusions = predictor.predict(text, caption)

        print(f"\n【第{i+1}条】")
        print(f"真实标签: {'真实' if label else '虚假'}")
        print(f"子结论: {sub_conclusions}")
        print(f"最终预测: {'真实' if pred_label else '虚假'}")

        if pred_label == label:
            correct += 1
            if label == 1:
                correct_real += 1
            else:
                correct_fake += 1

    # === 结果输出 ===
    total = len(texts)
    print("\n==== 结果统计 ====")
    print(f"总样本数: {total}")
    print(f"总准确率: {correct / total:.4f}")
    # print(f"真新闻准确率: {correct_real / total_real:.4f}")
    # print(f"假新闻准确率: {correct_fake / total_fake:.4f}")
    if total_real > 0:
        print(f"真新闻准确率: {correct_real / total_real:.4f}")
    else:
        print("真新闻准确率: 无（数据中没有真新闻样本）")

    if total_fake > 0:
        print(f"假新闻准确率: {correct_fake / total_fake:.4f}")
    else:
        print("假新闻准确率: 无（数据中没有假新闻样本）")


if __name__ == '__main__':
    main()
