from flask import Flask, request, jsonify, render_template
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)
model_path = r"C:\Users\moham\Desktop\Pr4(Question Generation)\Model"
tokenizer_path = r"C:\Users\moham\Desktop\Pr4(Question Generation)\Tokenizer"
# تحميل النموذج
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']

        # تحويل النص إلى مدخلات النموذج
        inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

        # إجراء التنبؤ
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=100)

        # تحويل المخرجات إلى نص
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({'output_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)