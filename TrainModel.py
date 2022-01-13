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

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, mode):
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        if mode == "age":
            target = target[:,0]
            target = target.unsqueeze(1).to(torch.float32)
        elif mode == "ethnicity":
            target = target[:,1]
        elif mode == "gender":
            target = target[:,2]

        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        if mode != "age":
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # Count correct predictions overall 
            pred = pred.reshape(-1)
            correct += torch.sum(pred == target)
        else:
            # pred = torch.round(output)
            correct += torch.sum(abs(output - target) <= 1)
    
    train_loss = float(np.mean(losses))

    # if mode != "age":
    train_acc = 100 * correct / ((batch_idx+1) * batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(float(np.mean(losses)), correct, (batch_idx+1) * batch_size,train_acc))
    return train_loss, train_acc

    


def test(model, device, criterion, test_loader, mode):
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample

            if mode == "age":
                target = target[:,0]
                target = target.unsqueeze(1).to(torch.float32)
            elif mode == "ethnicity":
                target = target[:,1]
            elif mode == "gender":
                target = target[:,2]

            data, target = data.to(device), target.to(device)
            
            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute loss based on same criterion as training
            loss = criterion(output,target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            if mode != "age":
                # Get predicted index by selecting maximum log-probability
                pred = output.argmax(dim=1, keepdim=True)
                
                # Count correct predictions overall 
                pred = pred.reshape(-1)
                correct += torch.sum(pred == target)
            else:
                # pred = torch.round(output)
                correct += torch.sum(abs(output - target) <= 1)

    test_loss = float(np.mean(losses))

    # if mode != "age":
    test_acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), test_acc))
    return test_loss, test_acc
    

def run_main():
    gender_name = ["Male","Female"]
    ethnicity_name = ["White","Black","Asian","Indian","Other"]
    df_age = pd.DataFrame(columns = ["train_loss","train_accuracy","test_loss","test_accuracy"])
    df_gender = pd.DataFrame(columns = ["train_loss","train_accuracy","test_loss","test_accuracy"])
    df_ethnicity = pd.DataFrame(columns = ["train_loss","train_accuracy","test_loss","test_accuracy"])

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Define parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4
    weight_decay = 1e-5

    # Initialize the model and send to device 
    model_age = ConvNet("age").to(device)
    model_gender = ConvNet("gender").to(device)
    model_ethnicity = ConvNet("ethnicity").to(device)
    
    # Define loss function.
    cross_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()

    # Define optimizer function.

    optimizer_age = optim.Adam(model_age.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_gender = optim.Adam(model_gender.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_ethnicity = optim.Adam(model_ethnicity.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Load datasets for training and testing
    df = pd.read_csv("age_gender.csv")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)
    picture_df = test_df.sample(n=20, random_state=12)
    picture_df_x = picture_df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32")).apply(lambda x:  x.reshape(1,48,48)).values
    picture_df_y = picture_df.iloc[:, 0:3].values

    train_dataset = PeopleDataset(train_df)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = PeopleDataset(test_df)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print("\nAge")
    # Train Age
    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        print(f'current epoch {epoch}')
        train_loss, train_accuracy = train(model_age, device, train_dataloader, optimizer_age, regression_criterion, epoch, batch_size, "age")
        test_loss, test_accuracy = test(model_age, device, regression_criterion, test_dataloader, "age")
        df_age = df_age.append({"train_loss" : train_loss, "train_accuracy" : float(train_accuracy), "test_loss" : test_loss, "test_accuracy" : float(test_accuracy)}, ignore_index = True)
        if test_loss < best_loss:
            best_loss = test_loss
            print("saving model")
            torch.save(model_age.state_dict(), "model_age.pth")

    print()

    print("Gender")
    # Train Gender
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f'current epoch {epoch}')
        train_loss, train_accuracy = train(model_gender, device, train_dataloader, optimizer_gender, cross_criterion, epoch, batch_size,"gender")
        test_loss, test_accuracy = test(model_gender, device, cross_criterion, test_dataloader,"gender")
        df_gender = df_gender.append({"train_loss" : train_loss, "train_accuracy" : float(train_accuracy), "test_loss" : test_loss, "test_accuracy" : float(test_accuracy)}, ignore_index = True)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print("saving model")
            torch.save(model_gender.state_dict(), "model_gender.pth")
    print()

    print("Ethnicity")
    # Train Ethnicity
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f'current epoch {epoch}')
        train_loss, train_accuracy = train(model_ethnicity, device, train_dataloader,optimizer_ethnicity, cross_criterion, epoch, batch_size,"ethnicity")
        test_loss, test_accuracy = test(model_ethnicity, device, cross_criterion, test_dataloader,"ethnicity")
        df_ethnicity = df_ethnicity.append({"train_loss" : train_loss, "train_accuracy" : float(train_accuracy), "test_loss" : test_loss, "test_accuracy" : float(test_accuracy)}, ignore_index = True)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print("saving model")
            torch.save(model_ethnicity.state_dict(), "model_ethnicity.pth")
    
    # write data to csv
    path_age = "./Data/age_results.csv"
    path_gender = "./Data/gender_results.csv"
    path_ethnicity = "./Data/ethnicity_results.csv"
    df_age.to_csv(path_age,index=False)
    df_gender.to_csv(path_gender,index=False)
    df_ethnicity.to_csv(path_ethnicity,index=False)

    # load best models
    model_age.load_state_dict(torch.load("model_age.pth"))
    model_gender.load_state_dict(torch.load("model_gender.pth"))
    model_ethnicity.load_state_dict(torch.load("model_ethnicity.pth"))

    # set models in eval mode
    model_age.eval()
    model_gender.eval()
    model_ethnicity.eval()
    
    # perform inference
    print("Training and evaluation finished")
    tbp = np.zeros((20,1,48,48))
    for i in range(len(picture_df)):
            tbp[i][0] = (picture_df_x[i][0])
    tbp = torch.tensor(tbp).to(device).float()
    results = []
    results.append(np.rint(model_age(tbp).reshape(-1).to("cpu").detach().numpy()))
    results.append(model_gender(tbp).argmax(dim=1, keepdim=True).reshape(-1).to("cpu").numpy())
    results.append(model_ethnicity(tbp).argmax(dim=1, keepdim=True).reshape(-1).to("cpu").numpy())

    plt.figure(figsize=(40,8))
    plt.gray()

    # plot inference
    for i in range(len(picture_df_x)):
        plt.subplot(2,10,i+1)
        plt.imshow(picture_df_x[i][0],cmap="gray")
        plt.title("Actual " + str(picture_df_y[i][0]) + " " + ethnicity_name[picture_df_y[i][1]] + " " + gender_name[picture_df_y[i][2]] + "\n"
        + "Results " + str(int(results[0][i])) + " " + ethnicity_name[results[2][i]] + " " + gender_name[results[1][i]], fontdict={'fontsize': 16})
        # plt.show()
    plt.savefig("Results/results.jpg")
    
    
if __name__ == '__main__':
    run_main()
    
    