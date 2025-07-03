from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# 模型和处理器加载
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

# 加载图像
image = Image.open("E:/HyZhen/pythonProject/CloudProject/datas/twitter_dataset/devset/images/boston_fake_35.jpeg").convert("RGB")

# 生成描述
# inputs = processor(image, text="a detailed description of the image", return_tensors="pt").to("cuda")
inputs = processor(image, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=64)
caption = processor.decode(out[0], skip_special_tokens=True)

print("图像描述：", caption)
