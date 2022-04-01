# run by typing python3 main.py in a terminal 
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from utils import get_base_url, allowed_file, and_syntax
from flask_cors import cross_origin


import pandas as pd
import numpy as np
from keras.models import load_model
from numpy import array
import os
from PIL import Image
from pickle import dump, load
import pickle
from time import time
from keras.preprocessing import sequence
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import cv2




model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)
print("Model Running...")


# In[4]:


def preprocess(image_path):
    path = image_path
    img=cv2.imread(image_path)
    img=cv2.resize(img, (299,299))
#     x = image.img_to_array(img)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x



def encode(image):
    image = preprocess(image)
    fea_vec = model_new.predict(image)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec




vocab = pd.read_csv("vocab.csv",names=[0])
vocab = np.array(vocab)


# In[8]:


# Giving index no. to vocabulary of each words
ixtoword = {}
wordtoix = {}
ix = 1
for word in vocab:
    ixtoword[ix] = word[0]
    wordtoix[word[0]] = ix
    ix += 1
print(ixtoword[1])
print(wordtoix["startseq"])



model_1 = load_model("model_31.h5")



def greedySearch(photo):
    in_text = 'startseq'
    max_length = 34
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model_1.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


# In[11]:


import matplotlib.pyplot as plt
def predict(path):
    encoding_train = {}
    encoding_train[path] = encode(path)
#     encoding_train
    pic = list(encoding_train.keys())[0]
    image = encoding_train[pic].reshape((1,2048))
    x=plt.imread(pic)
    plt.imshow(x)
    plt.show()
    return greedySearch(image)




# setup the webserver
'''
    coding center code
    port may need to be changed if there are multiple flask servers running on same server
    comment out below three lines of code when ready for production deployment
'''
port = 10009
base_url = get_base_url(port)
app = Flask(__name__, static_url_path=base_url+'static')


UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False
    
    

#@app.route('/')
@app.route(base_url)
def home():
    return render_template('home.html')

#@app.route('/', methods=['POST'])
@app.route(base_url, methods=['POST'])
def home_post():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('results', filename=filename))
    
    if "filesize" in request.cookies:
        if not allowed_image_filesize(request.cookies["filesize"]):
            print("Filesize exceeded maximum limit")
            return redirect(request.url)
    
    
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)
#         


#@app.route('/uploads/<filename>')
@app.route(base_url + '/uploads/<filename>')
def results(filename): 
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    res = predict(image_path)
    return render_template('results.html', filename=filename, labels = res)
    
    

       

#@app.route('/files/<path:filename>')
@app.route(base_url + '/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)




if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'coding.ai-camp.dev'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    # remove debug=True when deploying it
    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)

    '''
    cv scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)

