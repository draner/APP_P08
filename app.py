from flask import Flask, render_template, request, flash
import numpy as np
import re
import os
import pandas as pd
from collections import namedtuple
import cv2
import base64
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG_SIZE = 192
NUM_CLASSES = 8

# Read the list of images
img_names = []
mask_names = []

for root, dirs, files in os.walk('static/img/val_mask'):
    for file in files:
        if file.endswith('labelIds.png'):
            mask_names.append(os.path.join(root, file))

for root, dirs, files in os.walk('static/img/val_img'):
    for file in files:
        if file.endswith('leftImg8bit.png'):
            img_names.append(os.path.join(root, file))

# On crée un dataframe avec les chemins vers les images et les masques
df = pd.DataFrame({'img': img_names, 'mask': mask_names})

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label',['name','id','trainId','category','categoryId','hasInstances','ignoreInEval','color',])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

mapping_cat = {l.id : l.categoryId for l in labels}

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_SIZE, IMG_SIZE))
    x = x.astype(np.float32) / 255.0
    return x

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    label_mask = np.zeros_like(mask)
    for k in mapping_cat:
        label_mask[mask == k] = mapping_cat[k] 
    label_mask = np.eye(NUM_CLASSES)[label_mask]
    return label_mask

def prediction(img):
    image = cv2.imread(img)
    # encode image as base64 string
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes)
    # send request
    json_content = {"image": img_base64.decode('utf-8')}
    response = requests.post("https://api-p8.herokuapp.com/", json=json_content)
    # decode the mask
    mask = base64.b64decode(response.json()["mask"])
    mask = np.frombuffer(mask, dtype=np.float32)
    mask = mask.reshape(192, 192, 8)
    return mask

# Functions

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'some_secret'
@app.route('/predict')

def index():
    flash("Inconnu", 'prediction')
    flash("", 'tweet')
    return render_template('index.html', img_list=df['img'].values)

@app.route('/predicted', methods=['POST', 'GET'])
def predicted():
    if request.method == 'POST':
        path_img = request.form['option']
        index_df = df[df['img'] == path_img].index[0]
        path_mask = df['mask'][index_df]
        img = read_image(path_img)
        mask = read_mask(path_mask)
        predicted_mask = prediction(path_img)
        plt.subplot(1, 3, 1)
        plt.title('image')
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.title('mask')
        plt.imshow(np.argmax(mask, axis=-1))
        plt.subplot(1, 3, 3)
        plt.title('predicted mask')
        plt.imshow(np.argmax(predicted_mask, axis=-1))
        plt.savefig('static/predicted.png')
        flash(path_img, 'prediction')
        flash(path_mask, 'prediction')
        flash("static/predicted.png", 'image')
        return render_template('index.html', img_list=df['img'].values)