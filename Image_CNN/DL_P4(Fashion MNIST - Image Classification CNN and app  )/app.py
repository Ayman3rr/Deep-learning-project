from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model_path = r'D:\Project_githup\ma\DL_P4(Fashion MNIST - Image Classification CNN )\trained_fashion_mnist_model.h5'
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
                img = img.convert('L')  # Ensure the image is in grayscale
                img = img.resize((28, 28))
                img_array = np.array(img)
                img_array = img_array.reshape((1, 28, 28, 1))
                img_array = img_array / 255.0

                result = model.predict(img_array)
                predicted_class = np.argmax(result)
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                prediction = class_names[predicted_class]
                return render_template('index.html', prediction=prediction)
            except Exception as e:
                return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
