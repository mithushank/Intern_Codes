import torch
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.optim import SGD
from torchvision import models
from torchvision.models import resnet18,vgg19,resnet50,resnet34
from torchvision import transforms
from torchvision.transforms import ToTensor,ToPILImage
from torch.utils.data.dataloader import DataLoader

from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, CLStream51, SplitCIFAR100,PermutedMNIST
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin,ReplayPlugin
from avalanche.models import MobilenetV1, SimpleMLP, MTSlimResNet18, SimpleCNN,slda_resnet,SlimResNet18
from avalanche.training import MER, ICaRL,AGEM
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.training import RandomExemplarsSelectionStrategy, HerdingSelectionStrategy,FeatureBasedExemplarsSelectionStrategy
from avalanche.training.storage_policy import ParametricBuffer,ExemplarsBuffer,ExperienceBalancedBuffer
import numpy
import matplotlib.pyplot as plt

from functools import reduce
from typing import Union
       

benchmark = SplitCIFAR10(n_experiences=5,seed=1234,train_transform=transforms.ToTensor(),eval_transform=transforms.ToTensor())
# Load the pre-trained ResNet18 model
model = MTSlimResNet18(nclasses=10)



# Define SGD optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
# Define Criterion CrossEntrophy
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True,stream=True),
    loggers=[logger]
)
replay_plugin = ReplayPlugin(
    mem_size=1000,
    batch_size=100,
    storage_policy=ParametricBuffer(selection_strategy=RandomExemplarsSelectionStrategy(),max_size=1000)
)   
# Define MER strategy
strategy = AGEM(
    model = model,
    patterns_per_exp=5,
    optimizer = optimizer,
    criterion= criterion,
    plugins=[eval_plugin,replay_plugin],
    train_epochs=1,
    train_mb_size=64
    
)

for batch in (benchmark.train_stream):
    strategy.train(batch)    
    strategy.eval(benchmark.test_stream)
