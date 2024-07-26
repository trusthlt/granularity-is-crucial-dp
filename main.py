from settings import Settings, parse_arguments
from experiments import *
import numpy as np
import logging
import random
import torch
import jax

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    # get the settings from the command line
    settings = Settings(parse_arguments()).args
    np.random.seed(settings.seed)
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed(settings.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    logger.info(f"Dataset: {settings.dataset}")
    logger.info(f"Model: {settings.model}")
    logger.info(f"Seed: {settings.seed}")
    logger.info(f"Count device: {jax.local_device_count()}")
    if not settings.generate:
        logger.info(f"Lang Pair: {settings.lang_pair}")

        if settings.num_train_examples is not None:
            logger.info(f"Train on {settings.num_train_examples} training examples")
        else:
            logger.info(f"Train on full training data")

        if settings.num_eval_examples is not None:
            logger.info(f"Eval on {settings.num_eval_examples} eval examples")
        else:
            logger.info(f"Eval on full eval data")
    
        logger.info(f"Evaluate on test set: {settings.test}")

        logger.info(f"Epochs: {settings.epochs}")
        logger.info(f"Train batch size on 1 device: {settings.train_batch_size}")
        logger.info(f"Train batch size on all devices: {settings.train_batch_size * jax.local_device_count()}")
        logger.info(f"Eval batch size: {settings.eval_batch_size}")
        logger.info(f"Gradient accumulation steps: {settings.gradient_accumulation_steps}")
        logger.info(
            f"Gradient accumulation train batch size on all devices: "
            f"{settings.gradient_accumulation_steps * settings.train_batch_size * jax.local_device_count()}")
        logger.info(f"Optimizer: {settings.optimizer}")
        logger.info(f"Learning rate: {settings.learning_rate}")
        logger.info(f"Input sequence length: {settings.input_max_seq_len}")
        logger.info(f"Output sequence length: {settings.output_max_seq_len}")

        logger.info(f"Early stopping: {settings.early_stopping}")
        if settings.early_stopping:
            logger.info(f"Patience: {settings.patience}")
            logger.info(f"Minimum delta between updates: {settings.early_stop_min_delta}")

        if settings.private:
            logger.info(f"Private training")
            logger.info(f"L2 norm clip: {settings.l2_norm_clip}")
            logger.info(f"Noise multiplier: {settings.noise_multiplier}")
        else:
            logger.info(f"Normal training inf epsilon")

    # Load
    logger.info(f"Loading experiment")

    if "mt5" in settings.model:
        experiment = MT5Experiment(settings)
    elif "mlong" in settings.model:
        experiment = MLongT5Experiment(settings)
    elif "t5" in settings.model:
        experiment = T5Experiment(settings)
    else:
        raise ValueError("Model is not supported")

    experiment()
    if settings.generate:
        experiment.run_generate()
    else:
        experiment.run_experiment()


if __name__ == '__main__':
    main()
