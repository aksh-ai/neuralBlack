import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from flask import Flask, flash, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
LABELS = ['Meningioma', 'Glioma', 'Pitutary']

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "m4xpl0it"

device_name = "cuda:0:" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

resnet_model = models.resnet50(pretrained=True)

for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features

resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.4),
                nn.Linear(2048, 2048),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.4),
                nn.Linear(2048, 3),
                nn.LogSoftmax(dim=1))

for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

resnet_model.to(device)

resnet_model.load_state_dict(torch.load('models\\bt_total_resnet_torch.pt'))

resnet_model.eval()

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

                    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    img = transform(img)

                    img = img[None, ...]

                    if device_name=="cuda:0:":
                        img = img.cuda()

                    with torch.no_grad():
                        y_hat = resnet_model.forward(img)

                        predicted = torch.max(y_hat.data, 1)[1] 

                        print(LABELS[predicted.data])

                        session['pred_label'] = LABELS[predicted.data]
                        session['filename'] = filename

                        return redirect(url_for('pred_page'))

            else:
                print("Only POST requests are welcomed.")

    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return render_template('index.html', task=None)

if __name__=="__main__":
    app.run(debug=True)