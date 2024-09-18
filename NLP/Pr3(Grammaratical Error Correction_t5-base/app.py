from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# تحميل النموذج
model_name = r"C:\Users\moham\Desktop\Pr3(Grammaratical Error Correction_t5-base\Model"
tok_name = r"C:\Users\moham\Desktop\Pr3(Grammaratical Error Correction_t5-base\Tokenizer"

model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(tok_name)


# التحقق من وجود GPU واستخدامه إذا كان متاحًا
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# دالة لتصحيح النص
def correct_text(text, model, tokenizer, max_length=512):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    corrected_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return corrected_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['text']
        corrected_text = correct_text(input_text, model, tokenizer)
        return render_template('index.html', input_text=input_text, corrected_text=corrected_text)
    return render_template('index.html', input_text='', corrected_text='')

if __name__ == "__main__":
    app.run(debug=True)
