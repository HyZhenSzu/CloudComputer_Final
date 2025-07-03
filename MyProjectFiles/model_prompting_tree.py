class TreePromptPredictor:
    def __init__(self, glm_model, tokenizer):
        self.model = glm_model
        self.tokenizer = tokenizer

    def predict(self, text, image_caption):
        """
        输入文本和图像描述，返回最终真假判断（0=假新闻，1=真新闻）和子结论列表
        """
        sub_conclusions = []

        # 层级 1：基础判断
        sub_conclusions.append(self.ask_level_1(text, image_caption))

        # 层级 2：反事实假设——图像误导性
        sub_conclusions.append(self.ask_level_2(text))

        # 层级 3：反事实假设——文本夸大
        sub_conclusions.append(self.ask_level_3(text))

        # 最终归纳判断
        final = self.aggregate(sub_conclusions)

        return final, sub_conclusions

    def ask_level_1(self, text, image_caption):
        prompt = (
            "请阅读以下内容并判断其真实性：\n"
            f"【文本】{text}\n"
            f"【图像描述】{image_caption}\n\n"
            "请判断：这条新闻是真实还是虚假？只回答“真实”或“虚假”。"
        )
        return self._query(prompt)

    def ask_level_2(self, text):
        prompt = (
            "设想以下图片是与文本无关或旧图，请重新判断文本本身是否为真实新闻：\n"
            f"{text}\n\n"
            "如果只看文本，你觉得它是真的还是假的？只回答“真实”或“虚假”。"
        )
        return self._query(prompt)

    def ask_level_3(self, text):
        prompt = (
            "假设这条文本在表达时存在夸大或隐瞒，你会认为它可能是假新闻吗？\n"
            f"{text}\n\n"
            "请基于这种假设重新判断，回答“真实”或“虚假”。"
        )
        return self._query(prompt)

    def aggregate(self, conclusions):
        """
        多数投票方式决策：虚假>=真实 即判定为假新闻（0），否则为真新闻（1）
        """
        label_map = {"虚假": 0, "真实": 1}
        label_counts = {0: 0, 1: 0}

        for resp in conclusions:
            for k in label_map:
                if k in resp:
                    label_counts[label_map[k]] += 1
                    break

        final = 0 if label_counts[0] >= label_counts[1] else 1
        return final

    # def _query(self, prompt):
    #     response, _ = self.model.chat(
    #         self.tokenizer,
    #         prompt,
    #         history=[],
    #         max_new_tokens=64
    #     )
    #     return response.strip()

    def _query(self, prompt: str) -> str:
        try:
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=[],
                max_new_tokens=64
            )
            return response
        except Exception as e:
            print(f"[错误] 模型调用失败：{e}")
            return "模型回答失败"
