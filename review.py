import torch
from torchvision.models import *
from dataset.dataset import CompetitionDataset
from torchvision import transforms
from toolsp.train_test import *
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SHAPE = (240, 240)

# Create the model
model = resnext50_32x4d(num_classes=2)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1) # Just for practical reasons

# Load preprocess transformations
preprocess = transforms.Compose([
	transforms.Lambda(lambda x: x.convert('RGB')),
	transforms.Resize(SHAPE),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load competition set
# Please put the competition dataset folder just above CWD
competition_dataset = CompetitionDataset("../dataset/competition_test/", preprocess, device)
competition_loader = DataLoader(competition_dataset, batch_size=40, shuffle=False)



### ON MODEL SUBMISSION 4
# Load the weights
model.load_state_dict(torch.load('./saved_models/sub_4.pt'))

# Infere on competition dataset
results_sub_4 = competition(model, competition_loader, device, optimizer)



### ON MODEL SUBMISSION 5
# Load the weights
model.load_state_dict(torch.load('./saved_models/sub_5.pt'))

# Infere on competition dataset
results_sub_5 = competition(model, competition_loader, device, optimizer)


### Aggregate both results
results = []
for i in range(len(results_sub_4)):
	results.append(results_sub_4[i] or results_sub_5[i])

print(results)

# Write to file
submission_path = './review_submission.txt'
with open(submission_path, 'a') as fw:
	fw.write(str(results[0]))
	for r in results[1:]:
		fw.write('\n' + str(r))