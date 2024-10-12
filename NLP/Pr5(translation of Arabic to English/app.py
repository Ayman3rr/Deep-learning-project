from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# تحميل النموذج والتوكنيزر
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\moham\Desktop\Pr5(translation of Arabic to English\Tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained(r"C:\Users\moham\Desktop\Pr5(translation of Arabic to English\model")

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        # ترميز النص المدخل
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        # توليد الترجمة
        outputs = model.generate(input_ids)
        # فك ترميز النص المترجم
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return render_template("index.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)
