# import
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

names = ["age","ethnicity","gender"]
for i in range(3):
    model = names[i]
    # read
    df = pd.read_csv("./Data/" + model +"_results.csv")

    # Acurracy graph
    plt.plot(df["train_accuracy"], label = "train accuracy")
    plt.plot(df["test_accuracy"], label = "test accuracy")
    plt.title("Model " + model +" Accuracy")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(prop={'size': 6})
    plt.xticks(np.arange(0, len(df), 1.0))
    plt.ylim([0,100])
    plt.savefig("./Graphs/" + model + "_Accuracy.jpg")
    plt.show()

    # Loss graph
    plt.plot(df["train_loss"], label = "train loss")
    plt.plot(df["test_loss"], label = "test loss")
    plt.title("Model " + model +" Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend(prop={'size': 6})
    plt.xticks(np.arange(0, len(df), 1.0))
    plt.savefig("./Graphs/" + model + "_Loss.jpg")
    plt.show()