import random

import wandb

wandb.init(entity="poludmik", project="wandb-in-docker-test")
for _ in range(100):
    wandb.log({"test_metric": random.random()})

