import io
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from flask import Flask, jsonify, request

app = Flask(__name__)
LABELS = ['None', 'Meningioma', 'Glioma', 'Pitutary']

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet_model = models.resnet50(pretrained=True)

for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 4),
                                nn.LogSigmoid())

for name, child in resnet_model.named_children():
    for name2, params in child.named_parameters():
        params.requires_grad = True

resnet_model.to(device)
resnet_model.load_state_dict(torch.load('models\\bt_resnet50_model.pt'))
resnet_model.eval()

def preprocess_image(image_bytes):
  transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
  img = Image.open(io.BytesIO(image_bytes))
  return transform(img).unsqueeze(0)

def get_prediction(image_bytes):
  tensor = preprocess_image(image_bytes=image_bytes)
  y_hat = resnet_model(tensor.to(device))
  class_id = torch.argmax(y_hat.data, dim=1)
  return str(int(class_id)), LABELS[int(class_id)]

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    img_bytes = file.read()
    class_id, class_name = get_prediction(img_bytes)
    return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
  app.run()
