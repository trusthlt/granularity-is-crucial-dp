from flags_attack import AttackSettings, parse_arguments
from attack_experiments import *
import numpy as np
import logging
import random
import torch
import jax


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    settings = AttackSettings(parse_arguments()).args
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed(settings.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    experiment = AttackExperiment(settings)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
