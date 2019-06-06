from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/cnnfirst_demo.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')
Label_list = ['Clothing_Sets_&_Variety_Packs',
 'Protective_Swim_Tops',
 'Skorts',
 'Socks',
 'Pins_Brooches',
 'Protective_Active_Bibs',
 'Clothing_&_Footwear_Variety_Packs',
 'Protective_Active_Pants',
 'Bandanas_Handkerchiefs_Pocket_Squares',
 'Protective_Active_Button-Downs',
 'Bra_Accessories',
 'Non-Brim_Hats',
 'Protective_Active_Bodysuits',
 'Cardigans_Kimonos_Wraps',
 'Corsets_Basques_Waist_Cinchers',
 'Pantyhose_Tights',
 'Everyday_Dress_Footwear',
 'Specialty_Sports_Tops',
 'Specialty_Sport_Footwear',
 'Sweaters',
 'Socks_Undergarments_Variety_Packs',
 'Overalls',
 'Pajama_Shorts',
 'Skirts',
 'Pajama_Pants',
 'Shoe_Add-Ons_Replacement_Parts',
 'Ties',
 'Earmuffs',
 'Headbands',
 'Button-Downs',
 'Camis_Strapless_Tops',
 'Bras',
 'Specialty_Sports_Bottoms',
 'One-Piece_Swimsuits',
 'Everyday_Dress_Bodysuits',
 'Belts',
 'Everyday_Dress_Jumpsuits',
 'Night_Dresses_&_Shirts',
 'Polos',
 'Neck_Gaiters',
 'Active_Athletic_Footwear',
 'Sweatshirts_Fleece_Pullovers',
 'Swim_Bottoms',
 'Jackets_Coats',
 'Tie_Accessories',
 'Everyday_Dress_Shorts',
 'Specialty_Sports_Gloves',
 'Semi-Brim_Hats',
 'Full_Brim_Hats',
 'Balaclavas_Hoods',
 'Swim_Variety_Packs',
 'Shoe_Care',
 'Protective_Active_Vests',
 'One-Piece_Undergarments',
 'Scarves_&_Shawls',
 'Protective_Active_Shorts',
 'Eye_Masks',
 'Arm_Leg_Sleeves',
 'Swim_Tops',
 'Dresses_&_Gowns',
 'Gloves_Mittens',
 'Protective_Footwear',
 'Everyday_Dress_Vests',
 'Pajama_Jumpsuits',
 'Slacks_Pants',
 'Breast_Covers',
 'Robes',
 'Blazers_Suit_Coats',
 'Swim_Cover_Ups',
 'Cufflinks_&_Studs',
 'Protective_Active_Tops',
 'Business_Formal_Dress_Suits',
 'Jeans',
 'Legwarmers',
 'Shoe_Horns',
 'Accessories_Variety_Packs',
 'Protective_Active_Jumpsuits',
 'Suspenders',
 'Leggings',
 'Slips_Bloomers',
 'Badges_Patches']

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(50, 50))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict_proba(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds.shape)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = np.squeeze(preds)   # ImageNet Decode
        top_values_index=sorted(range(len(pred_class)), key=lambda i: pred_class[i])[-3:]
        top_values=[pred_class[i] for i in np.argsort(pred_class)[-3:]]
        top_values_index.reverse()
        top_values_index = [Label_list[r] for r in top_values_index]
        top_values.reverse()
            #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
            #result = str(pred_class[0][0][1])               # Convert to string
        preds = {top_values_index[0]:top_values[0],
                top_values_index[1]:top_values[1],
                top_values_index[2]:top_values[2],
                }      
        # result = str(pred_class[0][0][1])               # Convert to string
        return str(preds)
    return None



if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
