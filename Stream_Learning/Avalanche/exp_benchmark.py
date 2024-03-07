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
        
benchmark = SplitMNIST(n_experiences=5, seed=1234)


# def feature_based_selector(benchmark):
# list_of_selected_examples = []
for i,batch in enumerate(benchmark.train_stream):
    dataset,_ = batch.dataset,batch.task_label
    d = []
    sel = HerdingSelectionStrategy( FeatureBasedExemplarsSelectionStrategy(model=slda_resnet,layer_name='layer4.2'))
    s = sel.make_sorted_indices_from_features()
    # s = FeatureBasedExemplarsSelectionStrategy.make_sorted_indices('self',strategy=AGEM, data=dataset)
    # mnist_ava = data_incremental_benchmark(batch,experience_size=5000)
    # print(len(mnist_ava),len(D))

# egt the data points from the ex[perience] and select through the Random selector
def random_selector(benchmark):
    list_of_selected_examples_from_exp = []
    for i,batch in enumerate(benchmark.train_stream):
        dataset,_ = batch.dataset,batch.task_label
        d = []
        #Defining the buffer size
        buffer_size = 5000   # tunable parameter
        
        # random selector
        s = RandomExemplarsSelectionStrategy.make_sorted_indices('self',strategy=AGEM,data=dataset)
        count = 0
        for i in s:
            if count < buffer_size:
                d.append(dataset[i] )
                count+=1
        
        batch.dataset = d
        list_of_selected_examples_from_exp.append(batch)
        # list_of_selected_examples_from_exp.extend(batch)
    return list_of_selected_examples_from_exp

