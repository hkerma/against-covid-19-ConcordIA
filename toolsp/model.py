import torch 
from torchvision.models import *

class Model(torch.nn.Module):

	def __init__(self):
		torch.nn.Module.__init__(self)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.model1 = resnext50_32x4d(num_classes=2)
		self.model1.to(device)

		self.model2 = resnext50_32x4d(num_classes=2)
		self.model2.to(device)
		
	def forward(self, x):
		y1 = self.model1(x)
		y2 = self.model2(x)

		_, preds1 = torch.max(y1, 1)
		_, preds2 = torch.max(y2, 1)

		results = []

		for i in range(len(preds1)):
			results.append(int(torch.logical_or(preds1[i], preds2[i]) == True))

		return(results)