from transformers import AutoModelForImageTextToText, AutoProcessor
import os
import uuid
import base64
import argparse
from PIL import Image
from termcolor import colored

######################## Flask
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/share/project/zhouenshen/hpfs/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
######################## Flask

def decode_base64_to_file(base64_str, prefix="image"):
    filename = f"{UPLOAD_FOLDER}/{prefix}_{uuid.uuid4().hex}.png"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(base64_str))
    return filename


# NOTE: Your Qwen3-VL model path.
model_name = "/share/project/zhouenshen/hpfs/ckpt/vlm/Qwen3-VL-8B-Instruct"

model = AutoModelForImageTextToText.from_pretrained(
    model_name, dtype="auto", device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_name)


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()

    image_urls = data.get("image_url", [])
    text = data.get("text", "")

    image_files = [decode_base64_to_file(img_b64, prefix="image") for img_b64 in image_urls]
    image_list = [Image.open(os.path.join(UPLOAD_FOLDER, image_file)).convert('RGB') for image_file in image_files]

    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    for img in image_list:
        messages[0]["content"].append({
            "type": "image",
            "image": img
        })
    
    messages[0]["content"].append({
        "type": "text",
        "text": text
    })

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(colored(output_text, "cyan", attrs=["bold"]))

    for img_f in image_files:
        os.remove(img_f)

    response = jsonify({'result': 1, 'answer': output_text})

    response.headers.set('Content-Type', 'application/json')

    return response



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=25557)
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)