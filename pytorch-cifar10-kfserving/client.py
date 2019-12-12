import requests
import time
import torch
import torchvision
import torchvision.transforms as transforms
    


def test_model():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./model/data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    images, _ = dataiter.next()

    request = {"instances" : images[0:1].tolist()}
    responses = []

    start = time.time()
    for i in range(1000):
        responses.append(requests.post("http://localhost:8080/v1/models/cifar10:predict", json=request).json())
    print(len(responses))
    end = time.time()

    print("%f seconds elapsed for 1000 requests (%d RPS)" % (end - start, (1000.0 / (end - start))))

if __name__ == '__main__':
    test_model()
