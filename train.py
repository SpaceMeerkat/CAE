import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import FITSCubeDataset
import numpy as np
from sklearn.metrics import accuracy_score
from model import Net, weight_init
from torchvision import transforms
import os


def train(model: torch.nn.Module, transforms, data_path="./EXAMPLE_CUBES", num_epochs=50, batch_size=16, verbose=True,
          cube_length=640, img_size=(64, 64), loss=torch.nn.MSELoss()):

    data_path = os.path.abspath(data_path)

    model = model.train()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device).to(torch.float)

    loader = DataLoader(FITSCubeDataset(data_path, cube_length, transforms, img_size), batch_size, shuffle=True,
                        num_workers=4)

    optim = torch.optim.Adam(model.parameters())

    for i in range(num_epochs):
        print("Epoch %d of %d" % (i+1, num_epochs))

        accuracies = []
        for idx, (batch, target) in enumerate(tqdm(loader)):
            batch, target = batch.to(device).to(torch.float), target.to(device).to(torch.float).unsqueeze(-1)

            pred = model(batch)

            loss_value = loss(pred, target)

            optim.zero_grad()
            loss_value.backward()
            optim.step()

            pred_npy = pred.detach().cpu().numpy()
            target_npy = target.detach().cpu().numpy()

            pred_int = np.round(pred_npy).astype(np.uint8).reshape(-1)
            target_npy = target_npy.astype(np.uint8).reshape(-1)

            accuracies.append(accuracy_score(target_npy, pred_int)*100)

        mean_accuracy = sum(accuracies)/len(accuracies)
        print("Mean accuracy: %f" % mean_accuracy)


if __name__ == '__main__':
    model = Net()
    model.apply(weight_init)
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([10], [1])])

    train(model, transforms, num_epochs=50, batch_size=32)




