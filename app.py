import os
from flask import Flask, render_template, request, url_for
from PIL import Image
import numpy as np
import keras



# Load your brain tumor detection model
model = keras.models.load_model('Brain_tumour_Detection.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        uploaded_file = request.files['file']
        # Get patient's name and age from the form
        name = request.form['full-name']
        age = request.form['age']


        if uploaded_file.filename != '':
            # Save the uploaded file to a temporary directory
            image_path = os.path.join('static', 'uploads', uploaded_file.filename)
            uploaded_file.save(image_path)

            # Load and preprocess the image
            image = Image.open(image_path).convert('L')
            image = image.resize((256, 256))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)
           

            # Make predictions
            prediction = model.predict(image)[0][0]

            if prediction < 0.5:  
                predicted_class = "Healthy"
            else:
                predicted_class = "Tumor Detected"

            return render_template('index1.html', name=name, age=age , prediction=predicted_class, image_path=image_path)
        else:
            return "No file uploaded. Please try again."

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return f"Error processing image: {str(e)}. Please try again."

if __name__ == '__main__':
    app.run(debug=True)
