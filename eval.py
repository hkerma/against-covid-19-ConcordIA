import torch
from torchvision.models import *
from dataset.dataset import CompetitionDataset
from torchvision import transforms
from toolsp.train_test import *
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from toolsp.model import Model

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--model", type=str, default="saved_models/model.pt")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SHAPE = (240, 240)

PATH_TO_IMAGES = ["/home/hossein.aboutalebi/data/challenge/competition_test/{}.png".format(id) for id in range(1, 401)]
print(PATH_TO_IMAGES)
PATH_TO_MODEL = args.model



# Create the model
model = Model()
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

### ON MODEL SUBMISSION
# Load the weights
model.load_state_dict(torch.load(PATH_TO_MODEL))

# Infere on competition dataset
results_sub = competition(model, competition_loader, device, optimizer)

print(results_sub)