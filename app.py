import os
from flask import Flask, request, render_template,send_from_directory
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
import matplotlib.pyplot as plt
from pickle import dump, load
app = Flask(__name__)
# tokenizer = Tokenizer()
# Set the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_features(filename, model):
    try:
      image = Image.open(filename)
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
      if index == integer:
          return word
  return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        print(tokenizer.texts_to_sequences([in_text]))
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'photo' not in request.files:
        return "No file part"

    file = request.files['photo']

    if file.filename == '':
        return "No selected file"

    if file:
        # filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        print(filename)

        # Perform image captioning on the uploaded image
        xception_model = Xception(include_top=False, pooling="avg")
        photo = extract_features(filename, xception_model)
        img = Image.open(filename)

        description = generate_desc(model, tokenizer, photo, max_length)

        # Return the image caption as a response
        image_url = f"/uploads/{file.filename}"
        # Render the 'result.html' template with the image URL and description
        return render_template('result.html', image_url=image_url, description=description)

    return "File upload failed"

if __name__ == '__main__':
    max_length = 32
    # Load the model using Keras
    # model = tf.keras.models.load_model('model_9.h5')
    tokenizer = load(open("tokenizer.p","rb"))
    # model = tf.keras.models.load_model('model_9.h5', compile=False)
    model = load_model('model_9.h5',compile=False)
    # model = load_model('model_9.h5')
    app.run(debug=True)