from flask import Flask, render_template, request, jsonify
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load tokenizer and model from Hugging Face model hub
model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

max_input_length = 128
max_target_length = 128

def preprocess_function(input_text):
    # Tokenize the input text
    tokenized = tokenizer([input_text], return_tensors='tf', padding='max_length', truncation=True, max_length=max_input_length)
    return tokenized

def postprocess_prediction(prediction):
    # Decode the prediction
    return tokenizer.decode(prediction[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.json.get('text')
    
    # Preprocess the input text
    model_inputs = preprocess_function(input_text)
    
    # Make predictions
    output_sequences = model.generate(**model_inputs, max_length=max_target_length)
    
    # Postprocess the prediction
    translated_text = postprocess_prediction(output_sequences)
    
    return jsonify({'translation': translated_text})


if __name__ == '__main__':
    app.run(debug=True)
