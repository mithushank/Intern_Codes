import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18,vgg19,resnet50
from avalanche.benchmarks.classic import SplitFMNIST, SplitCIFAR10,SplitCIFAR100
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ICaRL

from torchvision import models


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_name='resnet18'):
        super(FeatureExtractor, self).__init__()
        
        # Load the specified pretrained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
        # Add more options for other models as needed
        
        # Replace the last fully connected layer with an identity layer
        if 'resnet' in model_name:
            self.model.fc = torch.nn.Identity()
        elif 'vgg' in model_name:
            self.model.classifier[-1] = torch.nn.Identity()
        # Add similar modifications for other models if required

    def forward(self, x,dataset = 'SplitMNIST'):
        # Convert grayscale images to RGB by duplicating the single channel
        '''
        For the different types of datasets we need to adjust the number of channels
        for SplitMNIST: (1,3,1,1)
        for SplitCIFAR10: (1,1,1,1)
        for SplitCIFAR100: (1,1,1,1)
        '''
        if dataset=='SplitCIFAR10':
            x = x.repeat(1, 1, 1, 1)     
        elif dataset==('SplitMNIST' or 'SplitFMNIST'):
            x = x.repeat(1, 3, 1, 1)     
        return self.model(x)


# Define classifier model
class Classifier(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# Load SplitFMNIST dataset
benchmark = SplitFMNIST(n_experiences=5, seed=1234)
benchmark_2 = SplitCIFAR10(n_experiences=5,seed=1234)
benchmark_3 = SplitCIFAR100(n_experiences=5,seed=1234)

# Define ResNet18-based feature extractor
feature_extractor = FeatureExtractor(model_name='resnet50')
feature_extractor_d = FeatureExtractor(model_name='resnet18')
feature_extractor_dd = FeatureExtractor(model_name='vgg16')

# Define SGD optimizer
optimizer = SGD(feature_extractor.parameters(), lr=0.001, momentum=0.9)
optimizer_d = SGD(feature_extractor_d.parameters(), lr=0.001, momentum=0.9)
optimizer_dd =  SGD(feature_extractor_dd.parameters(), lr=0.001, momentum=0.9)
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
strategy = ICaRL(
    feature_extractor=feature_extractor,
    classifier=Classifier(input_size=2048, num_classes=100),
    optimizer=optimizer,
    memory_size=100,
    buffer_transform=None,
    fixed_memory=True,
    train_epochs=2,
    train_mb_size=1024,
    plugins=[eval_plugin]
)
strategy_d = ICaRL(
    feature_extractor=feature_extractor_d,
    classifier=Classifier(input_size=512, num_classes=10),
    optimizer=optimizer_d,
    memory_size=100,
    buffer_transform=None,
    fixed_memory=True,
    train_epochs=2,
    train_mb_size=32,
    plugins=[eval_plugin]
)

strategy_dd = ICaRL(
    feature_extractor=feature_extractor_dd,
    classifier=Classifier(input_size=512, num_classes=10),
    optimizer=optimizer,
    memory_size=100,
    buffer_transform=None,
    fixed_memory=True,
    train_epochs=2,
    train_mb_size=128,
    plugins=[eval_plugin]
)

# Train the model

for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print('Model used to trained: Resnet50')
    
    strategy.train(experience)


    print("Computing accuracy on the whole test set")
    print('Model used to trained: Resnet50')
    strategy.eval(benchmark.test_stream)

