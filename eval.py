import torch 
from PIL import Image
_ = torch.manual_seed(42)
import PIL
from torchmetrics.multimodal import CLIPScore
from transformers import AutoImageProcessor, Dinov2Model, CLIPProcessor, CLIPModel
import numpy as np
import torch.fft as fft 
import torch.nn.functional as F
import glob 
from torchvision import transforms

from transformers import (
	CLIPTokenizer,
	CLIPTextModelWithProjection,
	CLIPVisionModelWithProjection,
	CLIPImageProcessor,
)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir = './huggingface_models/')
dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base", cache_dir = './huggingface_models/').cuda()
sscd_model = torch.jit.load("").cuda() # Download pretrained model from https://github.com/facebookresearch/sscd-copy-detection
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './huggingface_models/').cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './huggingface_models/')

clip_score = 0
sscd_score_total = 0 
dino_score_total = 0 
phase_score = 0 

data = open('', 'r')
data = data.readlines()

for data_point in data:
	print(data_point)
	data_point = data_point.split(';')
	prompt1 = data_point[3] # source prompt
	prompt2 = data_point[5] # driving prompt
	prompt3 = data_point[1] #target prompt

	source_path = './data/' + data_point[2]
	driving_path = './data/' + data_point[4]

	source_image = PIL.Image.open(source_path).convert("RGB")
	source_image = source_image.resize((512, 512)) 
	
	driving_image = PIL.Image.open(driving_path).convert("RGB")
	driving_image = driving_image.resize((512, 512)) 
	
	gen_images_path = ''
	gen_images = glob.glob(gen_images_path + '*.png')

	for i in gen_images:
		gen_image = PIL.Image.open(i).convert("RGB")
		gen_image = gen_image.resize((512, 512)) 
		
		# CLIP Score 

		clip_inputs = clip_processor(text=[prompt3], images=gen_image, return_tensors="pt", padding=True)
		clip_inputs['pixel_values'] = clip_inputs['pixel_values'].cuda()
		clip_inputs['input_ids'] = clip_inputs['input_ids'].cuda()
		clip_inputs['attention_mask'] = clip_inputs['attention_mask'].cuda()
		outputs = clip_model(**clip_inputs)
		logits_per_image = outputs.logits_per_image.abs().cpu().detach().numpy()
		clip_score = clip_score + logits_per_image

		# DINO Score
		
		image_inputs1 = image_processor(source_image, return_tensors="pt")
		image_inputs2 = image_processor(gen_image, return_tensors="pt")
		image_inputs1['pixel_values'] = image_inputs1['pixel_values'].cuda()
		image_inputs2['pixel_values'] = image_inputs2['pixel_values'].cuda()

		with torch.no_grad():
			outputs1 = dino_model(**image_inputs1)
			outputs2 = dino_model(**image_inputs2)

		last_hidden_states1 = outputs1.last_hidden_state # 1, 257, 768
		last_hidden_states2 = outputs2.last_hidden_state 

		dino_score = F.cosine_similarity(last_hidden_states1.view(-1), last_hidden_states2.view(-1), dim=0)
		dino_score_total = dino_score_total + dino_score
		#print("DINO Score is: ", dino_score)

		# SSCD Score

		image1_sscd = small_288(source_image).unsqueeze(0)
		image2_sscd = small_288(gen_image).unsqueeze(0)
		image1_sscd = image1_sscd.cuda()
		image2_sscd = image2_sscd.cuda()
		embedding1 = sscd_model(image1_sscd)[0, :]
		embedding2 = sscd_model(image2_sscd)[0, :]
		sscd_score = F.cosine_similarity(embedding1.view(-1), embedding2.view(-1), dim=0).abs().detach().cpu().numpy()
		sscd_score_total = sscd_score_total + sscd_score
		
		#print("SSCD Score is: ", sscd_score)

		# Phase score 

		image1 = np.array(driving_image)
		image1 = torch.from_numpy(image1)
		image1 = image1.permute(2,0,1)
		if(torch.max(image1)>2):
			image1 = image1/255.0

		image2 = np.array(gen_image)
		image2 = torch.from_numpy(image2)
		image2 = image2.permute(2,0,1)
		if(torch.max(image2)>2):
			image2 = image2/255.0


		image1 = torch.unsqueeze(image1, 0)
		driving_fft = fft.fft2(image1, dim=(2,3))
		driving_fft = driving_fft.angle()
		driving_fft = driving_fft.abs()
		image2 = torch.unsqueeze(image2, 0)
		result_fft = fft.fft2(image2, dim=(2,3))
		result_fft = result_fft.angle()
		result_fft = result_fft.abs()
		phase_diff = F.cosine_similarity(driving_fft.view(-1), result_fft.view(-1), dim=0)
		phase_score = phase_score + phase_diff
		
print("CLIP Score is: ", clip_score/(len(data)*5))
print("Pose Score is: ", phase_score/(len(data)*5))
print("DINO Score is: ", dino_score_total/(len(data)*5))
print("SSCD Score is: ", sscd_score_total/(len(data)*5))
