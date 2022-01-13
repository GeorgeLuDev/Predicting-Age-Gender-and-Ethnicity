import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class ConvNet(nn.Module):
    def __init__(self,mode):
        super(ConvNet, self).__init__()

        # base model
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1,32,5)
        self.b1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,5)
        self.b2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*9*9,1000) # 100
        self.b3 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000,100)
        self.b4 = nn.BatchNorm1d(100)
        self.output_layer_age = nn.Linear(100,1)
        self.output_layer_gender = nn.Linear(100,2)
        self.output_layer_ethnicity = nn.Linear(100,5)

        if mode == "age":
            self.forward = self.model_age
        elif mode == "gender":
            self.forward = self.model_gender
        elif mode == "ethnicity":
            self.forward = self.model_ethnicity
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    def model_age(self, X):
        # 1*48*48
        X = self.conv1(X) # 1*44*44
        X = self.b1(X)
        X = F.leaky_relu(X)
        X = F.max_pool2d(X,(2,2)) # 1*22*22

        X = self.conv2(X) # 1*18*18
        X = self.b2(X)
        X = F.leaky_relu(X)
        X = F.max_pool2d(X,(2,2)) # 1*9*9
        
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.b3(X)
        X = F.leaky_relu(X)
        X = self.fc2(X)
        X = self.b4(X)
        X = F.leaky_relu(X)
        X = self.output_layer_age(X)
        return  X

    def model_gender(self, X):
        # 1*48*48
        X = self.conv1(X) # 1*44*44
        X = self.b1(X)
        X = F.leaky_relu(X)
        X = F.max_pool2d(X,(2,2)) # 1*22*22

        X = self.conv2(X) # 1*18*18
        X = self.b2(X)
        X = F.leaky_relu(X)
        X = F.max_pool2d(X,(2,2)) # 1*9*9
        
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.b3(X)
        X = F.leaky_relu(X)
        X = self.fc2(X)
        X = self.b4(X)
        X = F.leaky_relu(X)
        X = self.output_layer_gender(X)
        return  X

    def model_ethnicity(self, X):
        # 1*48*48
        X = self.conv1(X) # 1*44*44
        X = self.b1(X)
        X = F.leaky_relu(X)
        X = F.max_pool2d(X,(2,2)) # 1*22*22

        X = self.conv2(X) # 1*18*18
        X = self.b2(X)
        X = F.leaky_relu(X)
        X = F.max_pool2d(X,(2,2)) # 1*9*9
        
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.b3(X)
        X = F.leaky_relu(X)
        X = self.fc2(X)
        X = self.b4(X)
        X = F.leaky_relu(X)
        X = self.output_layer_ethnicity(X)
        return  X