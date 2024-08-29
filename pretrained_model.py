import torch
import torchvision.transforms as transforms
from PIL import Image
from dinov2.models.vision_transformer import DinoVisionTransformer
from dinov2.models.vision_transformer import vit_base
from torchvision.transforms import ToTensor,ToPILImage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_config = {'img_size':64,
                'patch_size': 8, 
                'drop_path_rate': 0.3, 
                'init_values': 1e-05, 
                'drop_path_uniform': True, 
                'ffn_layer': 'mlp', 
                'block_chunks': 4, 
                'qkv_bias': True, 
                'proj_bias': True, 
                'ffn_bias': True, 
                'num_register_tokens': 0, 
                'interpolate_antialias': False, 
                'interpolate_offset': 0.1}
checkpoint_path = 'teacher_model_checkpoint.pth'
model = vit_base(**model_config)
print(model)
state_dict = torch.load(checkpoint_path)
print(f"Loading state using {checkpoint_path}")
model.load_state_dict(state_dict,strict=True)
model.to(device)

img_path = "n01443537_0.JPEG"
img = Image.open(img_path)

transform = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model.eval()

img_tensor = transform(img)

img_tensor = img_tensor.unsqueeze(0) # the shape of img_tensor is [1,3,224,224]

img_tensor = img_tensor.to(device)

with torch.no_grad():
    outputs = model(img_tensor) #the shape of outputs is [1,1024]

print(outputs.shape)