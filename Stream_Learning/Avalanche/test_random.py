import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18,vgg19,resnet50
from torchvision import transforms
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, CLStream51
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.models import MobilenetV1, SimpleMLP, MTSlimResNet18, SimpleCNN,slda_resnet
from avalanche.training import MER, ICaRL,AGEM
from avalanche.benchmarks.scenarios import CLStream, CLScenario, generic_scenario
from avalanche.benchmarks.classic.classic_benchmarks_utils import (
    check_vision_benchmark,
)
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.generators import dataset_benchmark, data_incremental_benchmark 
from avalanche.training import RandomExemplarsSelectionStrategy, HerdingSelectionStrategy,FeatureBasedExemplarsSelectionStrategy
from torchvision import models

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy



benchmark = SplitCIFAR10(n_experiences=5, seed=1234)
model = MTSlimResNet18(nclasses=10)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
# Define loss criterion
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[logger]
)

# Define ICaRL strategy
strategy = AGEM(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    patterns_per_exp=10
)



# def random_selector(benchmark):


list_of_selected_examples_from_exp = []
for i,batch in enumerate(benchmark.train_stream):
    dataset,_ = batch.dataset,batch.task_label
    d = []
    #Defining the buffer size
    buffer_size = 5000   # tunable parameter
    
    # print('train the batch before slection')
    # strategy.train(batch)
    # random selector
    s = RandomExemplarsSelectionStrategy.make_sorted_indices('self',strategy=AGEM,data=dataset)
    count = 0
    for i in s:
        if count < buffer_size:
            d.append(dataset[i] )
            count+=1
    batch.dataset = d
    list_of_selected_examples_from_exp.append(batch)
    print('Train after the selection of data')
    print('-------------------------------------------------------------------------------')
    strategy.train(batch)
    
    #cant able to train the dat as it says it couldnot train  the data in the list, but I extracted from  the list
    # before the selection I am able to train
    
        # list_of_selected_examples_from_exp.extend(batch)
    # return list_of_selected_examples_from_exp

