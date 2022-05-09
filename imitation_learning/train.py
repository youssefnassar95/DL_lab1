from __future__ import print_function

import sys

import torch

sys.path.append("../")

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from utils import *
from agent.bc_agent import BCAgent

from tensorboard_evaluation import Evaluation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1 - frac) * n_samples)], y[:int((1 - frac) * n_samples)]
    X_valid, y_valid = X[int((1 - frac) * n_samples):], y[int((1 - frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def one_hot_encoding(y, n_classes):
    one_hot_encode = []
    mapping = np.arange(0, n_classes)
    for i in range(len(y)):
        arr = list(np.zeros(n_classes, dtype=int))
        arr[mapping[y[i]]] = 1
        one_hot_encode.append(arr)

    return one_hot_encode


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):
    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    X_train = rgb2gray(X_train)
    X_train = np.expand_dims(X_train, axis=1)
    X_valid = rgb2gray(X_valid)

    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.
    y_train = list(map(action_to_id, y_train))
    y_valid = list(map(action_to_id, y_valid))

    y_train = one_hot_encoding(y_train, 4)
    y_train_straight = []
    X_train_straight = []
    y_train_left = []
    X_train_left = []
    y_train_right = []
    X_train_right = []
    y_train_accelerate = []
    X_train_accelerate = []
    # y_train_brake = []
    # X_train_brake = []


    for i in range(len(y_train)):
        if np.array_equiv(y_train[i], [1, 0, 0, 0]):
            y_train_straight.append(y_train[i])
            X_train_straight.append(X_train[i])
        if np.array_equiv(y_train[i], [0, 1, 0, 0]):
            y_train_left.append(y_train[i])
            X_train_left.append(X_train[i])
        if np.array_equiv(y_train[i], [0, 0, 1, 0]):
            y_train_right.append(y_train[i])
            X_train_right.append(X_train[i])
        if np.array_equiv(y_train[i], [0, 0, 0, 1]):
            y_train_accelerate.append(y_train[i])
            X_train_accelerate.append(X_train[i])
        # if np.array_equiv(y_train[i], [0, 0, 0, 0, 1]):
        #     y_train_brake.append(y_train[i])
        #     X_train_brake.append(X_train[i])

    X_train = [X_train_straight, X_train_left, X_train_right, X_train_accelerate]
    y_train = [y_train_straight, y_train_left, y_train_right, y_train_accelerate]

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    return X_train, y_train, X_valid, y_valid


def sample_minibatch(X, y, batch_size):
    small_batch = int(batch_size / 4)
    rand_list_straight = np.random.randint(0, high=len(X[0]), size=small_batch)
    rand_list_left = np.random.randint(0, high=len(X[1]), size=small_batch)
    rand_list_right = np.random.randint(0, high=len(X[2]), size=small_batch)
    rand_list_accelerate = np.random.randint(0, high=len(X[3]), size=small_batch)
    # rand_list_brake = np.random.randint(0, high=len(X[4]), size=small_batch)
    # rand_indeces = np.concatenate((rand_list_straight, rand_list_left, rand_list_right, rand_list_accelerate), axis=0)

    X_batch_straight = [X[0][i] for i in rand_list_straight]
    y_batch_straight = [y[0][i] for i in rand_list_straight]
    X_batch_left = [X[1][i] for i in rand_list_left]
    y_batch_left = [y[1][i] for i in rand_list_left]
    X_batch_right = [X[2][i] for i in rand_list_right]
    y_batch_right = [y[2][i] for i in rand_list_right]
    X_batch_accelerate = [X[3][i] for i in rand_list_accelerate]
    y_batch_accelerate = [y[3][i] for i in rand_list_accelerate]
    # X_batch_brake = [X[4][i] for i in rand_list_brake]
    # y_batch_brake = [y[4][i] for i in rand_list_brake]

    X_batch_concatenate = np.concatenate((X_batch_straight, X_batch_left, X_batch_right, X_batch_accelerate))
    y_batch_concatenate = np.concatenate((y_batch_straight, y_batch_left, y_batch_right, y_batch_accelerate))

    rand_list_batch = np.random.randint(0, high=batch_size, size=batch_size)
    X_batch = []
    y_batch = []
    for i in range(batch_size):
        X_batch.append(X_batch_concatenate[rand_list_batch[i]])
        y_batch.append(y_batch_concatenate[rand_list_batch[i]])

    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models",
                tensorboard_dir="./tensorboard"):
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    X_valid, y_valid = torch.from_numpy(X_valid).to(device), torch.from_numpy(y_valid).to(device)

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    agent = BCAgent(lr=lr)
    tensorboard_eval = Evaluation(tensorboard_dir, "CNN-4", stats=["train_acc", "valid_acc", "loss"])
    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # training loop
    for i in range(n_minibatches):

        X_batch, y_batch = sample_minibatch(X_train, y_train, batch_size)
        outputs, y_batch, loss = agent.update(X_batch, y_batch)

        if i % 10 == 0:
            _, predicted_train = torch.max(torch.abs(outputs).detach(), 1)
            # predicted_train.to(device)
            _, targetsbinary_train = torch.max(torch.abs(y_batch).detach(), 1)
            # targetsbinary_train.to(device)
            n_correct = (predicted_train == targetsbinary_train).sum().item()
            train_acc = n_correct * 100.0 / batch_size
            # train_acc = train_acc.item()
            print(f'train batch loss for iter {i}: {loss.item()}')
            print(f'batch accuracy: {train_acc} %')

            rand_valid = np.random.randint(0, len(y_valid), batch_size)
            X_valid = X_valid[rand_valid]
            y_valid = y_valid[rand_valid]

            # X_valid, y_valid = torch.from_numpy(X_valid).to(device), torch.from_numpy(y_valid).to(device)

            preds = agent.predict(X_valid.to(device))
            valid_acc = torch.sum(torch.argmax(preds, dim=-1) == y_valid) * 100.0 / batch_size
            valid_acc = valid_acc.item()
            # X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
            # y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
            #
            # predicted_valid = torch.argmax(torch.abs(agent.predict(X_valid)).detach(), 0)
            # # predicted_valid.to(device)
            # targetsbinary_valid = torch.argmax(torch.abs(y_valid).detach(), 0)
            # # targetsbinary.to(device)
            # n_correct = (predicted_valid == targetsbinary_valid).sum().item()
            # valid_acc = n_correct * 100.0 / batch_size
            # valid_acc = train_acc.item()
            print(f'valid accuracy: {valid_acc} %')

            eval = {"train_acc": train_acc, "valid_acc": valid_acc, "loss": loss}
            tensorboard_eval.write_episode_data(i, eval)
    print("finished")


    pass

    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent3.pt"))
    print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=1e-4)

