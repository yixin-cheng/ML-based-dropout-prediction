"""
This file is for the feature selection based on Genetic Algorithm
"""
# import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.cuda


# ## Step 2: Define settings
# 1. DNA size: the number of bits in DNA
# 2. Population size
# 3. Crossover rate
# 4. Mutation rate
# 5. Number of generations
class Selector:
    def __init__(self,df):
        self.df=df
    # define GA settings
        self.DNA_SIZE = len(self.df.columns)-1  # number of bits in DNA which equals the number of features
        self.POP_SIZE = 200  # population size
        self.CROSS_RATE = 0.75  # DNA crossover probability
        self.MUTATION_RATE = 0.002  # mutation probability
        self.N_GENERATIONS = 100  # generation size
        self.evolution()

    # ## Step 3: Define fitness, select, crossover, mutate functions

    def get_fitness(self,pop):
        """
        This function calculates the fitness (accuracy) in each DNA based on the Support Vector Machine algorithm
        :param pop: population
        :param path: the path is to the preprocessed data set
        :return: a list of accuracy of each DNA
        """
        res = []
        for element in pop:
            print(element)
            data = self.df
            data.drop(data.columns[0],axis=1)
            droplist = []
            for i in range(len(element)):
                if element[i] == 0:
                    droplist.append(i)
            data.drop(data.columns[droplist], axis=1)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            print('here')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                random_state=109)  # 80% training and 20% test
            clf = svm.SVC(kernel='linear')  # Linear Kernel

            # Train the model using the training sets
            clf.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(X_test)
            print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
            res.append(metrics.accuracy_score(y_test, y_pred))
        #     if torch.cuda.is_available():
        #         device = torch.device("cuda")
        #         print("Running on the GPU")
        #     else:
        #         device = torch.device("cpu")
        #         print("Running on the CPU")
        #
        #     # Hyper Parameters
        #     input_size = len(data.columns) - 1
        #     hidden_size = 50
        #     num_classes = 2
        #     num_epochs = 5
        #     batch_size = 10
        #     learning_rate = 0.01
        #
        #     class DataFrameDataset(torch.utils.data.Dataset):
        #         def __init__(self, df):
        #             self.data_tensor = torch.Tensor(df.values)
        #
        #         # a function to get items by index
        #         def __getitem__(self, index):
        #             obj = self.data_tensor[index]
        #             input = self.data_tensor[index][0:-1]
        #             target = self.data_tensor[index][-1]
        #
        #             return input, target
        #
        #         # a function to count samples
        #         def __len__(self):
        #             n, _ = self.data_tensor.shape
        #             return n
        #     msk = np.random.rand(len(data)) < 0.8
        #     train_data = data[msk]
        #     test_data = data[~msk]
        #
        #     # define train dataset and a data loader
        #     train_dataset = DataFrameDataset(df=train_data)
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=8)
        #
        #     """
        #     Step 2: Define a neural network
        #
        #     Here we build a neural network with one hidden layer.
        #         input layer: n neurons, representing the features of status of students
        #         hidden layer: 50 neurons, using Sigmoid/Tanh/Tanh/Softmax as activation function
        #
        #         output layer: 2 neurons, representing the type of dropouts
        #     """
        #
        #     # Neural Network
        #     class Net(nn.Module):
        #         def __init__(self, input_size, hidden_size, num_classes):
        #             super(Net, self).__init__()
        #             self.fc1 = nn.Linear(input_size, hidden_size)
        #             self.sigmoid = nn.Sigmoid()
        #             self.Tanh = nn.Tanh()
        #             self.Softmax = nn.Softmax()
        #             self.Relu = nn.ReLU()
        #             self.fc2 = nn.Linear(hidden_size, num_classes)
        #
        #         def forward(self, x):
        #             out = self.fc1(x)
        #             out = self.sigmoid(out)
        #             # out = self.Tanh(out)
        #             # out = self.Tanh(out)
        #             # out = self.Relu(out)
        #             out = self.fc2(out)
        #             return out
        #
        #     net = Net(input_size, hidden_size, num_classes).to(device=device)
        #
        #     # Loss and Optimizer
        #     criterion = nn.CrossEntropyLoss()
        #     optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
        #
        #     # store all losses for visualisation
        #     all_losses = []
        #
        #     # train the model by batch
        #     for epoch in range(num_epochs):
        #         total = 0
        #         correct = 0
        #         total_loss = 0
        #         for step, (batch_x, batch_y) in enumerate(train_loader):
        #             X = batch_x.to(device, non_blocking=True)
        #             Y = batch_y.long().to(device, non_blocking=True)
        #             # print(X)
        #             # Forward + Backward + Optimize
        #             optimizer.zero_grad()  # zero the gradient buffer
        #             outputs = net(X)
        #             loss = criterion(outputs, Y)
        #             all_losses.append(loss.item())
        #             loss.backward()
        #             optimizer.step()
        #
        #             if (epoch % 1 == 0):
        #                 _, predicted = torch.max(outputs, 1)
        #                 # calculate and print accuracy
        #                 total = total + predicted.size(0)
        #                 correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
        #                 total_loss = total_loss + loss
        #         if (epoch % 1 == 0):
        #             print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
        #                   % (epoch + 1, num_epochs,
        #                      total_loss, 100 * correct / total))
        #
        #     # Optional: plotting historical loss from ``all_losses`` during network learning
        #     # Please uncomment me from next line to ``plt.show()`` if you want to plot loss
        #
        #     # import matplotlib.pyplot as plt
        #     #
        #     # plt.figure()
        #     # plt.plot(all_losses)
        #     # plt.show()
        #
        #     """
        #     Evaluating the Results
        #
        #
        #     """
        #
        #     # train_input = train_data.iloc[:, :input_size]
        #     # train_target = train_data.iloc[:, input_size]
        #     #
        #     # inputs = torch.Tensor(train_input.values).float()
        #     # targets = torch.Tensor(train_target.values).long()
        #     #
        #     # outputs = net(inputs)
        #     # _, predicted = torch.max(outputs, 1)
        #     # total = predicted.size(0)
        #     # correct = predicted.detach() == targets.detach()
        #     test_input = test_data.iloc[:, :input_size]
        #     test_target = test_data.iloc[:, input_size]
        #
        #     inputs = torch.Tensor(test_input.values).float()
        #     targets = torch.Tensor(test_target.values - 1).long()
        #
        #     outputs = net(inputs)
        #     _, predicted = torch.max(outputs, 1)
        #
        #     total = predicted.size(0)
        #     correct = predicted.data.numpy() == targets.data.numpy()
        #
        #
        #     res.append(sum(correct) / total)
        #     print('Testing Accuracy: %.2f %%' % (100 * sum(correct) / total))
        return res


    # define population select function based on fitness value
    # population with higher fitness value has higher chance to be selected
    def select(self,pop, fitness):
        idx = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True,
                           p=fitness / sum(fitness))
        return pop[idx]


    # define gene crossover function
    def crossover(self,parent, pop):
        if np.random.rand() < self.CROSS_RATE:
            # randomly select another individual from population
            i = np.random.randint(0, self.POP_SIZE, size=1)
            # choose crossover points(bits)
            cross_points = np.random.randint(0, 2, size=self.DNA_SIZE).astype(np.bool)
            # produce one child
            parent[cross_points] = pop[i, cross_points]
        return parent


    # define mutation function
    def mutate(self,child):
        for point in range(self.DNA_SIZE):
            a = np.random.rand()
            if a < self.MUTATION_RATE:
                # print(a)
                child[point] = 1 if child[point] == 0 else 0
        return child


    # ## Step 4: Start training GA
    # 1. randomly initialise population
    # 2. determine fitness of population
    # 3. repeat
    #     1. select parents from population
    #     2. perform crossover on parents creating population
    #     3. perform mutation of population

    def evolution(self):
        """
        the whole process of genetic algorithm
        """
        # initialise population DNA
        pop = np.random.randint(0, 2, (self.POP_SIZE, self.DNA_SIZE))
        print(len(pop[0]))
        print(pop)
        for t in range(self.N_GENERATIONS):
            # train GA
            # calculate fitness value
            fitness = self.get_fitness(pop)  # translate each NDA into accuracy which is fitness
            # if the generation reaches the max, then abandon the bad performance feature and save the rest of features to a new file
            if t == self.N_GENERATIONS - 1:
                res = pop[np.argmax(fitness), :]
                print("Most fitted DNA: ", pop[np.argmax(fitness), :])
                data=self.df
                data.drop(data.columns[0], axis=1)
                droplist = []
                for i in range(len(res)):
                    if res[i] == 0:
                        droplist.append(i)
                print("Abandoned feature index: ", droplist)
                data=data.drop(data.columns[droplist], axis=1)
                if type(data) is not None:
                    self.df=data
            # select better population as parent 1
            pop = self.select(pop, fitness)
            # make another copy as parent 2
            pop_copy = pop.copy()

            for parent in pop:
                # produce a child by crossover operation
                child = self.crossover(parent, pop_copy)
                # mutate child
                child = self.mutate(child)
                # replace parent with its child
                parent[:] = child
        return self.df