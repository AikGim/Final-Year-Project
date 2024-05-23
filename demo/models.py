'''
this handles pytorch model loading for demo app
'''

import torch
import torch.nn as nn
import torch_pruning as tp

from torchvision.models import efficientnet_b0 as efficientnet
from efficientnet_lite import build_efficientnet_lite

device = "cuda" if torch.cuda.is_available() else "cpu"

def pytorch_effnet():
    model = efficientnet()
    in_feats = model.classifier[1].in_features
    dropout = model.classifier[0].p
    model.classifier = nn.Sequential(nn.Dropout(dropout, inplace = True), nn.Linear(in_features=in_feats, out_features=1, bias=True))

    for param in model.children():
        param.requires_grad_(False)

    model = model.to(device)
    model.eval()
    return model


def effnet_lite():
    model = build_efficientnet_lite('efficientnet_lite0', 1)

    for param in model.children():
        param.requires_grad_(False)

    model = model.to(device)
    model.eval()
    return model


# load model from saved state
def load_model_state(model, PATH):

    if "pruned" in PATH.lower():
        tp.load_state_dict(model, torch.load(PATH, map_location = torch.device('cpu')))
        
    else:
        model.load_state_dict(torch.load(PATH, map_location = torch.device('cpu')))

    model = model.to(device)
    model.eval()

    return model

def prepare_model(model_path):
    if "B0" in model_path.upper():
        model = pytorch_effnet()
        load_model_state(model, model_path)
    
    if "LITE" in model_path.upper():
        model = effnet_lite()
        load_model_state(model, model_path)
    
    return model

def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
    out = torch.sigmoid(logits)
    pred = (out > 0.5).float()
    
    return pred


if __name__ == '__main__':
    print("Using device: %s" % device)

    model = effnet_lite()

    load_model_state(model,"demo\models\LITE0-20pruned-5iter2-FT2.pth")

    print(model)