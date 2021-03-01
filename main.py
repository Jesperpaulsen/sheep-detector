import argparse

from torch.utils.data import random_split, DataLoader
import torchvision.models as models
import torch
from config import retinanet_config as config
from SheepDataset import SheepDataset
import numpy as np
import utils
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-a", "--annotations", default=config.OPTICAL_CSV,
                             help='path to annotations')
argument_parser.add_argument("-i", "--images", default=config.IMAGES_PATH,
                             help="path to images")
argument_parser.add_argument("-t", "--train", default=config.TRAIN_CSV,
                             help="path to output training CSV file")
argument_parser.add_argument("-e", "--test", default=config.TEST_CSV,
                             help="path to output test CSV file")
argument_parser.add_argument("-c", "--classes", default=config.CLASSES_CSV,
                             help="path to output classes CSV file")
argument_parser.add_argument("-s", "--split", type=float, default=config.TRAIN_TEST_SPLIT,
                             help="train and test split")
args = vars(argument_parser.parse_args())

annot_path = args['annotations']
images_path = args['images']
train_csv = args['train']
test_csv = args['test']
classes_csv = args['classes']
train_test_split = args['split']

dataset = SheepDataset(annot_path=annot_path, images_path=images_path)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
train_size = int(np.floor(len(dataset) * train_test_split))
test_size = int(len(dataset) - train_size)

train_set, test_set = random_split(dataset, [train_size, test_size])

data_loader_train = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
data_loader_test = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=0)
model = models.resnet152(pretrained=True)
model.to(device=device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.7, momentum=0.9, weight_decay=0.5)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1.5)
criterion = torch.nn.NLLLoss()

val_metrics = {
    'accuracy': Accuracy(),
    'nll': Loss(criterion),
}
trainer = create_supervised_trainer(model, optimizer, criterion)
evaluator = create_supervised_evaluator(model, metrics=val_metrics)


@trainer.on(Events.ITERATION_COMPLETED(every=1))
def log_training_loss(trainer):
    print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(data_loader_train)
    metrics = evaluator.state.metrics
    print(
        f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(data_loader_test)
    metrics = evaluator.state.metrics
    print(
        f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")


print('training')
trainer.run(data_loader_train, max_epochs=1)
