import numpy as np
from quiver_engine import server
from quiver_engine.model_utils import register_hook
from torchvision import  models
import torch
from unet import Unet
if __name__ == "__main__":
    model = Unet(1)
    Path='E:/Broswer/train_7m_Unet_change2_80.pth'
    model.load_state_dict(torch.load(Path,map_location=torch.device('cpu'))) 
    hook_list = register_hook(model)
    
    server.launch(model, hook_list, input_folder="./data/Cat", image_size=[512,512], use_gpu=False)

