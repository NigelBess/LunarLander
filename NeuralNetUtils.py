from typing import List
from typing import Callable
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class RawData[TState]:

    def __init__(self, state : TState, action: List[float], reward: float, nextState : TState):
        self.State = state
        self.Action = action# action is expected to be all zeros with a 1 for the chosen action
        self.Reward = reward
        self.NextState = nextState

class Layer(nn.Module):
    def __init__(self, nodeCount: int, activation_function: nn.Module):
        super().__init__()
        self.nodeCount = nodeCount
        self.activation_function = activation_function

    def setInputCount(self,inputCount:int):
        self.linear = nn.Linear(inputCount,self.nodeCount)
    
    def forward(self,x): return self.activation_function(self.linear(x))

    def set_to_cuda(self):
        self.linear = self.linear.cuda()
        self.activation_function = self.activation_function.cuda()

class NeuralNetwork(nn.Module):
    def __init__(self, input_feature_count:int, layers: nn.ModuleList):
        super().__init__()
        self.input_feature_count = input_feature_count
        lastNodeCount = input_feature_count
        for layer in layers:
            layer.setInputCount(lastNodeCount)
            lastNodeCount = layer.nodeCount

        self.layers = layers
            

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        for l in self.layers: x = l.forward(x)
        return x
    
    def set_to_cuda(self):
        for layer in self.layers:
            layer.set_to_cuda()

def buildLayers(hiddenLayers=[]):
    layers = nn.ModuleList()
    for i in hiddenLayers: layers.append(Layer(i,nn.ReLU()))
    layers.append(Layer(1,nn.Sigmoid()))
    return layers


def getPrediction(yHat:np.array): 
    return (yHat>0.5).astype(int)

def accuracy(prediction:np.array,target:np.array): #arrays should be 0 or 1
    return (sum(prediction==target)/len(prediction)).item()
    
def f1Score(prediction:np.array,target:np.array):#arrays should be 0 or 1
    actualPositives = sum(target)
    predictedPositives = sum(prediction)
    truePositives = sum(target*prediction)
    if actualPositives == 0: 
        recall = 0
    else:
        recall = truePositives/actualPositives

    if predictedPositives == 0:
        precision = 0
    else:
        precision = truePositives/predictedPositives
    if precision == 0 or recall == 0:
        return 0
    f1 = 2/(1/precision+ 1/recall)
    return f1


def convertToNumpy(tens:torch.tensor):
        if torch.cuda.is_available():
            tens = tens.cpu()
        return tens.detach().numpy()

class TrainingResult():
    def __init__(self):
        self.f1_score = []
        self.accuracy = []
        self.loss = []

class TrainingParams():
    def __init__(self):
        self.learningRate = 1e-2
        self.regularizationConstant = 1e-2
        self.iterations = 100



class DataSet:
    X = "X",
    Y = "Y",

    

    def __init__ (self, X : torch.tensor, Y : torch.tensor):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Row count mismatch between X and Y")

        #initialize self.datasets
        self.datasets = {DataSet.X: X, DataSet.Y: Y}       

        #make sure we can run gradient descent on x training data
        self.get(DataSet.X).requires_grad_(True)  

    @staticmethod
    def buildFromPandas( X: pd.DataFrame, Y: pd.DataFrame):
        def convert(set : pd.DataFrame):
            tens = torch.tensor(set.values, dtype=torch.float32)
            if len(set.shape)==1:
                tens.unsqueeze_(1)
            return tens

        #make sure we can run gradient descent on x training data
        return DataSet(X, Y, convert)
    
    @staticmethod
    def buildFromNumpy( X: np.array, Y: np.array):
        def convert(set : np.array):
            tens = torch.from_numpy(set)
            if len(set.shape)==1:
                tens.unsqueeze_(1)
            return tens

        #make sure we can run gradient descent on x training data
        return DataSet(X, Y, convert)


    @staticmethod
    def buildFromAny(X, Y, conversionToTensor : Callable):
        return DataSet(conversionToTensor(X), conversionToTensor(Y))
    

    
    @property
    def InputCount(self): return self.get(DataSet.X).shape[1]

    @property
    def Rows(self): return self.get(DataSet.X).shape[0]

    def assertSetExistence(self,setName):
        if setName not in self.datasets:
            raise ValueError(f"Dataset '{setName}' not found")
        
    def set(self, setName: str, value: torch.Tensor):
        self.assertSetExistence(setName)
        self.datasets[setName] = value
    
    def get(self, setName: str) -> torch.Tensor:
        self.assertSetExistence(setName)
        return self.datasets[setName]
    

    def set_to_cuda(self):
        with torch.no_grad():
            for setName in self.datasets:
                set = self.datasets[setName]
                if set is None: continue
                self.datasets[setName] = set.cuda()

def trainModel(dataSets:List[DataSet], model:NeuralNetwork, loss_function:nn.modules.loss._Loss, trainingParams:TrainingParams)->List[TrainingResult]: 
    trainData = dataSets[0]
    print(F"Training Model. Features {trainData.InputCount}, Samples {trainData.Rows}. Iterations: {trainingParams.iterations}")



    results:List[TrainingResult] = []
    for _ in dataSets:
        results.append(TrainingResult())

    optimizer = torch.optim.Adam(params=model.parameters(),lr=trainingParams.learningRate,weight_decay=trainingParams.regularizationConstant)

    if torch.cuda.is_available():
        for set in dataSets:
            set.set_to_cuda()
        model.set_to_cuda()
        print(F"Running on {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("Running on CPU")
    
    def convertOutput(y):
        return convertToNumpy(y.squeeze(1))

    for _ in range(trainingParams.iterations):
        trainingLoss = None
        for (set,result) in  zip(dataSets,results):
            x = set.get(DataSet.X)
            y = set.get(DataSet.Y)
            yHat = model.forward(x)

            l = loss_function(yHat,y)
            if trainingLoss is None: trainingLoss = l
            result.loss.append(l.item())

            y = convertOutput(y)
            yHat = convertOutput(yHat)
            prediction = getPrediction(yHat)

            acc = accuracy(prediction,y)
            result.accuracy.append(acc)

            f1 = f1Score(prediction,y)
            result.f1_score.append(f1)
        

        trainingLoss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"returning {len(results)} results for {len(dataSets)} data sets")
    return results