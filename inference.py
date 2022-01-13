from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from Model import ConvNet
from DataLoader import PeopleDataset
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2 as cv

def run_main():
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)

    model_age = ConvNet("age").to(device)
    model_gender = ConvNet("gender").to(device)
    model_ethnicity = ConvNet("ethnicity").to(device)

    model_age.load_state_dict(torch.load("model_age.pth"))
    model_gender.load_state_dict(torch.load("model_gender.pth"))
    model_ethnicity.load_state_dict(torch.load("model_ethnicity.pth"))

    model_age.eval()
    model_gender.eval()
    model_ethnicity.eval()
    
    gender_name = ["male","female"]
    ethnicity_name = ["white","black","asian","indian","other"]

    img = cv.imread("image.png",cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(48,48))
    tbp = np.zeros((1,1,48,48))
    tbp[0][0] = img
    tbp = torch.tensor(tbp).to(device).float()
    results = []
    results.append(np.rint(model_age(tbp).reshape(-1).to("cpu").detach().numpy()))
    results.append(model_gender(tbp).argmax(dim=1, keepdim=True).reshape(-1).to("cpu").numpy())
    results.append(model_ethnicity(tbp).argmax(dim=1, keepdim=True).reshape(-1).to("cpu").numpy())
    
    plt.imshow(img,cmap="gray")
    plt.title("predicted " + str(int(results[0][0])) + " " + ethnicity_name[results[2][0]] + " " + gender_name[results[1][0]])
    plt.savefig("image_predictions.jpg")
    plt.show()



if __name__ == '__main__':
    run_main()