from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model_path = r'C:\Users\moham\Desktop\DL_Pr6(Plantdisease_detection[PlantVillage]_app\Plant_model.h5'
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

            
                class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

                prediction = class_names[predicted_class]
                return render_template('index.html', prediction=prediction)
            except Exception as e:
                return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
