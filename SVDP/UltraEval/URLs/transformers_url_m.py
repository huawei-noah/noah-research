from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from gevent.pywsgi import WSGIServer
from URLs.dispatcher import GPUDispatcher as gdp
gdp.bind_worker_gpus()

app = Flask(__name__)

print("Initializing model and tokenizer...")

# Set the device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = os.environ.get('HF_MODEL_NAME')
port = os.environ.get('PORT')

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check and add pad token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

print("Model and tokenizer initialized.")

params_dict = {
    "do_sample": True,
    "temperature": 0.1,
    "max_new_tokens": 400,
    "top_p": 0.95,
}

@app.route('/infer', methods=['POST'])
def main():
    datas = request.get_json()
    params = datas["params"]
    prompt = datas["instances"]

    for key, value in params.items():
        if key == "max_tokens":
            params_dict["max_new_tokens"] = value
        elif key in params_dict:
            params_dict[key] = value
    if prompt == "":
        return jsonify({'error': 'No prompt provided'}), 400
    
    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)  # Prepare the input tensor
    generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **params_dict)

    # Decoding the generated ids to text
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert len(prompt) == len(generated_text)
    for j in range(len(prompt)):
        generated_text[j] = generated_text[j][len(prompt[j]):]
    return jsonify(generated_text)

if __name__ == '__main__':
    # Run the Flask app
    http_server = WSGIServer(('127.0.0.1', port), app)
    http_server.serve_forever()
