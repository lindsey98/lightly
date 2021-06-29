import torch
import torchvision
import lightly.models as models
import lightly.loss as loss
import lightly.data as data
import pytorch_lightning as pl
import math
import os
import shutil
from tqdm import tqdm
import logging
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

exp_name = 'CIFAR10'
start_epoch = 0
avg_loss = 0.
avg_output_std = 0.
epochs = 800
out_dim = 2048
input_size = 32

logging.basicConfig(filename='mytest_{}.log'.format(exp_name), level=logging.INFO)
logger = logging.getLogger('trace')


# the collate function applies random transforms to the input images
collate_fn = data.ImageCollateFunction(input_size=input_size, 
                                        # require invariance to flips and rotations?
                                        hf_prob=0.0, # horizontal flip prob
                                        vf_prob=0.0, # vertical flip prob
                                        rr_prob=0.0, # (+90 degree) rotation is applied prob
                                        min_scale=0.0,
                                        cj_prob=0.7, # color jitter prob
                                      )

# create a dataset from your image folder
dataset = data.LightlyDataset(input_dir='../datasets/{}/train/'.format(exp_name))
print('Dataset is loaded')
logger.info('Dataset is loaded')

# build a PyTorch dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,                # pass the dataset to the dataloader
    batch_size=64,         # a large batch size helps with the learning
    shuffle=True,           # shuffling is important!
    collate_fn=collate_fn, # apply transformations to the input images
    drop_last=True)  # FIXME: drop_last for distributed training, single-gpu training does not need this to be True

logger.info('Length of data {}'.format(len(dataloader.dataset)))

# use a resnet50 backbone
resnet = torchvision.models.resnet.resnet18()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# build the simsiam model
model = models.SimSiam(resnet, num_ftrs=512)

# use a criterion for self-supervised learning
criterion = loss.SymNegCosineSimilarityLoss()

# get a PyTorch optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-0, weight_decay=1e-5)

# push to device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.nn.DataParallel(model)

# if resume
if os.path.exists('../output/{}.pt'.format(exp_name)):
    model.load_state_dict(torch.load('../output/{}.pt'.format(exp_name), map_location="cpu"))
    logger.info('Resume model {}'.format(exp_name))
    
model.to(device)
print('Model is initialized and pushed to device')
logger.info('Model is initialized and pushed to device')

# Train!
for e in range(start_epoch, epochs):
    
    print('Epoch {}'.format(str(e)))
    logger.info('Epoch {}'.format(str(e)))
    
    for (x0, x1), _, _ in tqdm(dataloader):

        # move images to the gpu
        x0 = x0.to(device)
        x1 = x1.to(device)

        # run the model on both transforms of the images
        # the output of the simsiam model is a y containing the predictions
        # and projections for each input x
        y0, y1 = model(x0, x1)

        # backpropagation
        loss = criterion(y0, y1)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        output, _ = y0
        output = output.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        avg_loss = w * avg_loss + (1 - w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()
        
    torch.save(model.state_dict(), '../output/{}.pt'.format(exp_name))
    # the level of collapse is large if the standard deviation of the l2
    # normalized output is much smaller than 1 / sqrt(dim)
    collapse_level = max(0., 1 - math.sqrt(out_dim) * avg_output_std)
    # print intermediate results
    print(f'[Epoch {e:3d}] '
        f'Loss = {avg_loss:.2f} | '
        f'Collapse Level: {collapse_level:.2f} / 1.00')
    
    logger.info(f'[Epoch {e:3d}] '
        f'Loss = {avg_loss:.2f} | '
        f'Collapse Level: {collapse_level:.2f} / 1.00')
    
    
    