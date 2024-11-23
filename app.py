from flask import Flask, request, jsonify, render_template
import pickle
import os
from werkzeug.utils import secure_filename
from PIL import Image

# Define the Meso4 class
class Meso4:
    def __init__(self):
        self.description = "This is a placeholder for the Meso4 model."
    
    def predict(self, x):
        prediction_score = sum(sum(pixel) for pixel in x) % 2
        return "Fake" if prediction_score else "Real"

# Initialize Flask app
app = Flask(__name__)

# Directory for uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
try:
    with open('meso4_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the home route to render the HTML page
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deepfake Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f3f4f6;
            }
            .container {
                text-align: center;
                background: #fff;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            h2 {
                color: #333;
                margin-bottom: 20px;
            }
            form {
                display: inline-block;
            }
            input[type="file"] {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 20px;
                cursor: pointer;
            }
            button {
                background-color: #007BFF;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #0056b3;
            }
            .footer {
                margin-top: 20px;
                color: #777;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Upload an Image for Deepfake Detection</h2>
            <form action="/predict" method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required><br>
                <button type="submit">Submit</button>
            </form>
            <div class="footer"></div>
        </div>
    </body>
    </html>
    '''

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected image"}), 400

    # Save the uploaded image
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)

    try:
        # Preprocess the image
        img = Image.open(filepath)
        img = img.resize((224, 224))  # Resize to match model input
        img = img.convert('RGB')  # Ensure the image is in RGB format
        img_array = list(img.getdata())  # Example array extraction

        # Make prediction using the loaded model
        prediction = model.predict(img_array)

        # Return the prediction result
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #f3f4f6;
                }}
                .container {{
                    text-align: center;
                    background: #fff;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }}
                h3 {{
                    color: #333;
                    margin-bottom: 20px;
                }}
                a {{
                    text-decoration: none;
                    color: #007BFF;
                    font-size: 16px;
                }}
                a:hover {{
                    color: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h3>Prediction: {prediction}</h3>
                <a href="/">Go back</a>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
