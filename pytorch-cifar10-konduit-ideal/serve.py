import torch
from model.train import Model
import torchvision.transforms as transforms

transform = None
model = None


def setup():
    global transform
    global model

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    model = Model()
    model.load_state_dict(torch.load('./model/model.pt'))
    model.eval()


def run(request):
    global transform
    global model

    input_data = request['the_input'] # numpy array from the IMAGE_LOADINT step.
    input_tensor = transform(input_data).unsqueeze(0)
    return { 'the_prediction': model(input_tensor).detach().numpy() } # detach().numpy() sucks.
