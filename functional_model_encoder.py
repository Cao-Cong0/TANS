import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch.nn.functional as F
from config import *
from misc.utils import *
from ofa.imagenet_classification.elastic_nn.networks.ofa_resnets import OFAResNets
from ofa.imagenet_classification.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3
import torch.backends.cudnn as cudnn

def get_model_embedding(model_name,dataset,epoch):

    RESULT_PATH = "/home/cc/TANS/pretrained_result/"
    model_path = os.path.join(RESULT_PATH,f'{model_name}_{dataset}',f'{epoch}')

    # Set the random seed for reproducibility
    torch.manual_seed(40)
    torch.cuda.manual_seed_all(40)

    # Set the deterministic mode for the cuDNN backend
    cudnn.deterministic = True

    # Create an instance of the model
    if model_name == 'ResNet50':
        dim = 2048
        model = OFAResNets(n_classes=2, dropout_rate=0, depth_list=[
                           0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35], width_mult_list=[0.65, 0.8, 1.0])
    elif model_name == 'MobileNetV3_w1':
        dim = 1280
        model = OFAMobileNetV3(n_classes=2, dropout_rate=0, width_mult=1, ks_list=[
                               3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4])
    elif model_name == 'MobileNetV3_w1.2':
        dim = 1280
        model = OFAMobileNetV3(n_classes=2, dropout_rate=0, width_mult=1.2, ks_list=[
                               3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4])

    model.to('cuda')  # Move the model to the GPU

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Remove the last layer of the model
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # add a new layer to the model to get 1536-dim embedding
    model.classifier.add_module('new_layer', nn.Linear(dim, 1536))

    # feed fixed Gaussian noise as input to those models
    # to get the model embedding

    with torch.no_grad():
        # Move the noise tensor to the GPU
        noise = torch.randn(1, 3, 224, 224).cuda()
        model.to('cuda')
        model(noise)
        # print the model embedding
        print(model_name)
        print(model(noise).size())
        # save the model embedding
        np.save(model_name, model(noise).cpu().numpy())

    return model(noise).cpu().detach().numpy()


epoch_file = "epoch1.pt"
model_name = "ResNet50"
dataset = "pneumoniamnist2"
model_embedding = get_model_embedding(model_name,dataset,epoch_file)
print(model_embedding)









