import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import resnet18,vgg19,resnet50
from torchvision import transforms
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.models import MobilenetV1, SimpleMLP, MTSlimResNet18, SimpleCNN
from avalanche.training import MER

from torchvision import models
# class FeatureExtractor(torch.nn.Module):
#     def __init__(self, model_name='resnet18'):
#         super(FeatureExtractor, self).__init__()
        
#         # Load the specified pretrained model
#         if model_name == 'resnet18':
#             self.model = models.resnet18(pretrained=True)
#         elif model_name == 'resnet50':
#             self.model = models.resnet50(pretrained=True)
#         elif model_name == 'vgg16':
#             self.model = models.vgg16(pretrained=True)
#         # Add more options for other models as needed
        
#         # Replace the last fully connected layer with an identity layer
#         if 'resnet' in model_name:
#             self.model.fc = torch.nn.Identity()
#         elif 'vgg' in model_name:
#             self.model.classifier[-1] = torch.nn.Identity()
#         # Add similar modifications for other models if required

#     def forward(self, x):
#         # Convert grayscale images to RGB by duplicating the single channel
#         x = x.repeat(1, 3, 1, 1)
#         return self.model(x)


# # Define classifier model
# class Classifier(torch.nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(Classifier, self).__init__()
#         self.fc = torch.nn.Linear(input_size, num_classes)

#     def forward(self, x):
#         return self.fc(x)


# # Define ResNet18-based feature extractor
# feature_extractor = FeatureExtractor(model_name='resnet50')
# feature_extractor_d = FeatureExtractor(model_name='resnet18')
# feature_extractor_dd = FeatureExtractor(model_name='vgg16')

# Load SplitFMNIST dataset
benchmark = SplitMNIST(n_experiences=5,seed=1234)
# model 
model = SimpleMLP(num_classes=10 )
# Define SGD optimizer
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
# Define Criterion CrossEntrophy
criterion = CrossEntropyLoss()

# Define logger and evaluation plugin
logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[logger]
)

# Define MER strategy
strategy = MER(
    model = model,
    optimizer = optimizer,
    criterion= criterion,
    mem_size= 200,
    batch_size_mem= 200
)

# Train the model
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print('Model used to trained: MTSlimResnet18')
    strategy.train(experience)
 
    print("Computing accuracy on the whole test set")
    strategy.eval(benchmark.test_stream)

# import random

# class AvalancheDataset:
#     def __init__(self, data_stream, length):
#         self.data_stream = data_stream
#         self.length = length

#     def __len__(self):
#         return self.length

# class ExemplarsSelectionStrategy:
#     pass  # Placeholder for the base selection strategy class

# class RandomExemplarsSelectionStrategy(ExemplarsSelectionStrategy):
#     """Select the exemplars at random in the dataset"""

#     def make_sorted_indices(
#         self, strategy: "SupervisedTemplate", data: AvalancheDataset
#     ) -> list[int]:
#         indices = list(range(len(data)))
#         random.shuffle(indices)
#         return indices

# # Generate synthetic data stream
# def generate_data_stream(num_samples):
#     for i in range(num_samples):
#         # Generate synthetic data here, for example:
#         data_point = (random.random(), random.random())  # Example of a synthetic data point
#         yield data_point