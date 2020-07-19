import os
import requests
from flask import Flask, flash, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "m4xpl0it"

@app.route('/empty_page')
def empty_page():
    filename = session.get('filename', None)
    os.remove(os.path.join(UPLOAD_FOLDER, filename))
    return redirect(url_for('index'))

@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)

@app.route('/', methods=['POST', 'GET'])
def index():
    try:
        if request.method == 'POST':
            f = request.files['bt_image']
            filename = str(f.filename)

            if filename!='':
                ext = filename.split(".")
                
                if ext[1] in ALLOWED_EXTENSIONS:
                    filename = secure_filename(f.filename)

                    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename),'rb') as img:
                        predicted = requests.post("http://localhost:5000/predict", files={"file": img}).json()
                        print(predicted)

                    session['pred_label'] = predicted['class_name']
                    session['filename'] = filename

                    return redirect(url_for('pred_page'))

    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True, port=3000)