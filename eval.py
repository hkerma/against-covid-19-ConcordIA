import torch
from torchvision.models import *
from dataset.dataset import CompetitionDataset
from torchvision import transforms
from toolsp.train_test import *
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
<<<<<<< HEAD
from toolsp.model import Model

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--list", type=str, nargs="*")
parser.add_argument("--model", type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SHAPE = (240, 240)

PATH_TO_IMAGES = [r.strip("[],") for r in args.list]
print(PATH_TO_IMAGES)
PATH_TO_MODEL = args.model



# Create the model
model = Model()
=======

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--list", nargs="+")
parser.add_argument("--model4", type=str)
parser.add_argument("--model5", type=str)

args = parser.parse_args()
print(args.list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SHAPE = (240, 240)
PATH_TO_IMAGES = args.list
PATH_TO_MODEL_4 = args.model4
PATH_TO_MODEL_5 = args.model5

print(PATH_TO_MODEL_4)
print(PATH_TO_MODEL_5)

# Create the model
model = resnext50_32x4d(num_classes=2)
>>>>>>> abe632d9990ea9cc3c720e2eec1f78a5224afaef
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
competition_dataset = CompetitionDataset("../dataset/competition_test/", preprocess, device, PATH_TO_IMAGES)
competition_loader = DataLoader(competition_dataset, batch_size=1, shuffle=False)

<<<<<<< HEAD
### ON MODEL SUBMISSION
# Load the weights
model.load_state_dict(torch.load(PATH_TO_MODEL))

# Infere on competition dataset
results_sub = competition(model, competition_loader, device, optimizer)

print(results_sub)
=======


### ON MODEL SUBMISSION 4
# Load the weights
model.load_state_dict(torch.load(PATH_TO_MODEL_4))

# Infere on competition dataset
results_sub_4 = competition(model, competition_loader, device, optimizer)


### ON MODEL SUBMISSION 5
# Load the weights
model.load_state_dict(torch.load(PATH_TO_MODEL_5))

# Infere on competition dataset
results_sub_5 = competition(model, competition_loader, device, optimizer)


### Aggregate both results
results = []
for i in range(len(results_sub_4)):
	results.append(results_sub_4[i] or results_sub_5[i])

results

>>>>>>> abe632d9990ea9cc3c720e2eec1f78a5224afaef
