from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100,CORe50, SplitCUB200,CLStream51
from avalanche.benchmarks.generators import nc_scenario
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import MTSlimResNet18,SimpleMLP
from avalanche.training import RandomExemplarsSelectionStrategy, HerdingSelectionStrategy,Naive, ICaRL,AGEM,MIR,MER
from avalanche.training.plugins import EvaluationPlugin
from codecarbon import EmissionsTracker

"""
Need to include the bilevel selection and camel implementation
All datasets are included here
Need to add Learning scenarios: class incremetal, domain insremental
Need to add buffer management systems

"""

# Create a scenario with the  datasets
scenario_1 = SplitCIFAR10(
    n_experiences=5,
    seed=1234,  # for reproducibility
    train_transform=None,
    eval_transform=None
)
scenario_2 = SplitCIFAR100(
    n_experiences=50,
    seed=1234,
    shuffle=True,
    train_transform=None,
    eval_transform=None   
)

# Create a simple MLP model
model = MTSlimResNet18(num_classes=10)
model_1 = SimpleMLP

# Create an optimizer
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create a loss function
criterion = CrossEntropyLoss()

# Create logger and evaluation plugin
logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[logger]
)

# Initialize the CodeCarbon Emissions Tracker
tracker = EmissionsTracker()

# Start tracking
tracker.start()

# Create the Naive training strategy
strategy = Naive(
    model_1, optimizer, criterion, train_mb_size=32, train_epochs=1, eval_mb_size=32,
    plugins=[eval_plugin]
)

strategy_1 = RandomExemplarsSelectionStrategy(

)

strategy_2 = HerdingSelectionStrategy(

)

# Train and test the model on the scenario
for experience in scenario_1.train_stream:
    print("Start of experience: ", experience.current_experience)
    strategy.train(experience)
    print("End of experience: ", experience.current_experience)

    print("Computing accuracy on the whole test set")
    strategy.eval(scenario_1.test_stream)

# Stop tracking and display estimated emissions
emissions = tracker.stop()
print(f"Estimated emissions: {emissions} kg")
