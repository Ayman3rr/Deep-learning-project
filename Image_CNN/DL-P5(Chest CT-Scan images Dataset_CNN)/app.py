from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model_path = r'C:\Users\moham\Desktop\DL-P4(Chest CT-Scan images Dataset_CNN)\Chest CT-Scan images Dataset.h5'
model = tf.keras.models.load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            try:
                img = Image.open(io.BytesIO(file.read()))

                # التأكد من أن الصورة تحتوي على 3 قنوات لونية (RGB)
                img = img.convert('RGB')

                # تغيير حجم الصورة لتتناسب مع مدخلات النموذج
                img = img.resize((224, 224))

                # تحويل الصورة إلى مصفوفة numpy
                img_array = np.array(img)

                # تعديل الشكل ليصبح (1, 224, 224, 3)
                img_array = img_array.reshape((1, 224, 224, 3))

                # تطبيع القيم لتصبح بين 0 و 1
                img_array = img_array / 255.0

                result = model.predict(img_array)
                predicted_class = np.argmax(result)
                class_names = ['adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib', 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa', 'normal', 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa']
                prediction = class_names[predicted_class]
                return render_template('index.html', prediction=prediction)
            except Exception as e:
                return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
