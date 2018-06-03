import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import FITSCubeDataset
import numpy as np
from sklearn.metrics import accuracy_score
from model import RegressionNet, CategoricalNet, weight_init
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_accuracy(accuracies, epochs, filename):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(0, 100)
    plt.plot(epochs, accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epoch')

    fig.savefig(filename, bbox_inches='tight')
    plt.close()

def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    decay = initial_lr / num_epochs
    lr = initial_lr - decay*epoch
    print("Set LR to %f" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model: torch.nn.Module, transforms, data_path="./EXAMPLE_CUBES", num_epochs=50, batch_size=32, verbose=True,
          cube_length=640, img_size=(64, 64), loss=torch.nn.MSELoss(), lr_schedule=True, initial_lr=1e-3, suffix=""):

    data_path = os.path.abspath(data_path)
	
    model = model.train()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device).to(torch.float)

    loader = DataLoader(FITSCubeDataset(data_path, cube_length, transforms, img_size), batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    optim = torch.optim.Adam(model.parameters(), initial_lr)
	
    accuracies, epochs = [0], [0]
	
    for i in range(num_epochs):
        print("Epoch %d of %d" % (i+1, num_epochs))

        _accuracies = []
        for idx, (batch, target) in enumerate(tqdm(loader)):
            batch = batch.to(device).to(torch.float)
            if isinstance(loss, torch.nn.CrossEntropyLoss):
                target = target.to(device).to(torch.long)
            else:
                target = target.to(device).to(torch.float)
            pred = model(batch)

            loss_value = loss(pred, target)

            optim.zero_grad()
            loss_value.backward()
            optim.step()

            pred_npy = pred.detach().cpu().numpy()
            target_npy = target.detach().cpu().numpy()
			
            if isinstance(loss, torch.nn.CrossEntropyLoss):
                pred_npy = np.argmax(pred_npy, axis=1)
				
            pred_int = np.round(pred_npy).astype(np.uint8).reshape(-1)
            target_npy = target_npy.astype(np.uint8).reshape(-1)

            _accuracies.append(accuracy_score(target_npy, pred_int)*100)

        mean_accuracy = sum(_accuracies)/len(_accuracies)
        accuracies.append(mean_accuracy)
        epochs.append(i+1)
        if lr_schedule:
            plot_accuracy(accuracies, epochs, "accuracy_scheduler%s.png" % suffix)
            adjust_learning_rate(optim, i, initial_lr, num_epochs)
        else:
            plot_accuracy(accuracies, epochs, "accuracy_no_scheduler%s.png" % suffix)
        print("Mean accuracy: %f" % mean_accuracy)


if __name__ == '__main__':
    print("Creating Model and Initializing weights")
	
    for model_class, loss_fn, suffix in zip([CategoricalNet, RegressionNet], [torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()], ["_categorical", "_regression"]):
        for schedule in [True, False]:
    
            model = model_class()
            model.apply(weight_init)
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([10], [1])])

            train(model, transform, num_epochs=50, batch_size=64, lr_schedule=schedule, loss=loss_fn, suffix=suffix)




