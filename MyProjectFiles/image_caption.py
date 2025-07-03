import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class ImageCaptionGenerator:
    def __init__(self, model_path="Salesforce/blip-image-captioning-base", image_root=None):
        """
        :param model_path: Hugging Face模型路径或本地路径
        :param image_root: 存放图片的文件夹根目录
        """
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
        self.image_root = image_root

    def _load_image(self, image_path):
        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法读取图片 {image_path}，错误：{e}")
            return None

    def caption_image(self, image_path):
        image = self._load_image(image_path)
        if image is None:
            return ""

        print(f"Has loaded image: {image_path}")
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=64)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"图片描述：{caption}")
        return caption

    def caption_images_for_posts(self, image_id_lists: list[list[str]], sep=", ") -> list[str]:
        """
        为多个帖子生成图像描述文本（自动检测扩展名）。

        :param image_id_lists: 每个帖子对应的图片 ID 列表（无扩展名）。如：[['img1', 'img2'], ['img3']]
        :param sep: 多张图片之间连接的分隔符
        :return: 每个帖子的合并图像描述字符串列表
        """
        valid_extensions = [".jpg", ".jpeg", ".png", ".gif"]
        all_captions = []

        for post_idx, image_ids in enumerate(image_id_lists):
            if not image_ids:
                all_captions.append("该帖子没有图片")
                continue

            captions = []

            for idx, img_id in enumerate(image_ids):
                image_path = None
                for ext in valid_extensions:
                    potential_path = os.path.join(self.image_root, img_id + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break

                if image_path:
                    try:
                        caption = self.caption_image(image_path)
                        if caption:
                            captions.append(f"图片{idx + 1}描述：{caption}")
                        else:
                            captions.append(f"图片{idx + 1}无法生成描述")
                    except Exception as e:
                        print(f"⚠️ 图片加载错误：{image_path} - {str(e)}")
                        captions.append(f"图片{idx + 1}加载失败")
                else:
                    captions.append(f"图片{idx + 1}未找到（ID: {img_id}）")

            combined = sep.join(captions)
            all_captions.append(combined)

        return all_captions
