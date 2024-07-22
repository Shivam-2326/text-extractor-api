import cv2
from flask import Flask, request, jsonify
from pymongo import MongoClient
import pytesseract
from PIL import Image
import io
import base64
import numpy as np
from flask_cors import CORS

# Flask app
app = Flask(__name__)
CORS(app) 

# MongoDB client
client = MongoClient('mongodb://localhost:27017/')
db = client['image_text']
collection = db['extracted_texts']

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = image_file.read()
        base64_string = image_to_base64(image_bytes)

        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        d = pytesseract.image_to_data(
            thresh, output_type=pytesseract.Output.DICT)

        extracted_text = ""
        bold_text = ""

        n_boxes = len(d['level'])
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top']
                            [i], d['width'][i], d['height'][i])
            text = d['text'][i]
            if text.strip() != "":
                extracted_text += text + " "
                # Check if the text might be bold based on height/width ratio
                if w / h > 1.5:
                    bold_text += text + " "
       
        text_data = {
            'filename': image_file.filename,
            'text': extracted_text,
            'bold_text': bold_text,
            'image': base64_string,
            'content_type': image_file.content_type 
        }
        collection.insert_one(text_data)

        return jsonify({'message': 'Text has been extracted and stored in the database'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/images', methods=['GET'])
def get_images():
    try:
        # Retrieve all images from the MongoDB collection
        image_files = collection.find()

        images = []
        for item in image_files:
            image_data = {
                'filename': item['filename'],
                'text': item['text'],
                'bold_text': item['bold_text'],
                'image': item['image'],
                'content_type': item['content_type']
            }
            images.append(image_data)

        return jsonify(images)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def image_to_base64(image_bytes):
    base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
    return base64_encoded


if __name__ == '__main__':
    app.run(debug=True)
