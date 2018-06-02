import torch


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128 * 2 * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(256, 1))

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features.view(int(x.size()[0]), -1))
        # output= F.log_softmax(output,dim=1) # Give results using softmax
        return output