from VLM_demo import encode_image
from openai import OpenAI
import json
api_key= "sk-df55df287b2c420285feb77137467576"
base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
def _vlmapi_call(image_path):
    client = OpenAI(api_key=api_key,base_url=base_url)

    base64_image = encode_image(image_path)

    completion = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",  
        messages=[{"role": "user","content": [
                {"type": "text","text": f"This is a robotic arm operation scene. Detect all objects"},
                {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                }
                ]}]
        )

    resstr = completion.choices[0].message.content.replace("```","").replace("json","")

    state = json.loads(resstr)

    return state

