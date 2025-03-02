from openai import OpenAI
from PIL import Image, ImageOps, ImageDraw, ImageFont
import requests
import base64
from io import BytesIO


def encode_image(image):
    with BytesIO() as buffer:
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


image = Image.open('IMG_3055_JPG_jpg.rf.8f0a673a17c2b8d7af487b2d753ff337.jpg')
encoded_image = encode_image(image)
result = client.chat.completions.create(
    # model="export",
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "extract data in JSON format"},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
    ]}],
    temperature=0.1,
    top_p=0.7,
)
print("Completion result:", result)
# print result on image
message_content = result.choices[0].message.content
font = ImageFont.truetype('arial.ttf', 10)
draw = ImageDraw.Draw(image)
draw.text((20, 20), message_content,
          fill=(255, 0, 0), font=font)
filename = 'ocr_result.png'
image.save(filename)
print('save image: ', filename)
