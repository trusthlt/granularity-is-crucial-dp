import argparse


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')


def parse_arguments():
    arg_parser = argparse.ArgumentParser()

    # Main parameters
    arg_parser.add_argument("--dataset", type=str, required=True)
    arg_parser.add_argument(
        "--model",
        type=str,
        default='agemagician/mlong-t5-tglobal-base',
        help="Path to model"
    )
    arg_parser.add_argument("--save_path", type=str, default='checkpoints/')
    arg_parser.add_argument("--lang_pair", type=str, default='de-en')
    arg_parser.add_argument("--seed", type=int, default=666)
    arg_parser.add_argument(
        "--num_train_examples",
        type=int,
        default=None,
        help="Number of examples to select from train set. Without any input, select all data"
    )
    arg_parser.add_argument(
        "--num_eval_examples",
        type=int,
        default=None,
        help="Number of examples to select from dev set. Without any input, select all data"
    )
    arg_parser.add_argument(
        "--test",
        type=str2bool,
        nargs='?',
        const=True,
        default=False
    )
    # Main hyperparameters
    arg_parser.add_argument("--epochs", type=int, default=1)
    arg_parser.add_argument("--train_batch_size", type=int, default=8, help="Train batch size per device")
    arg_parser.add_argument("--eval_batch_size", type=int, default=32, help="Eval batch size per device")
    arg_parser.add_argument("--input_max_seq_len", type=int, default=64)
    arg_parser.add_argument("--output_max_seq_len", type=int, default=512)
    
    # Optimizer parameters
    arg_parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Optimizer either SGD or Adam, if private training, the DP version will be loaded automatically"
    )
    arg_parser.add_argument("--learning_rate", type=float, default=0.001)
    arg_parser.add_argument("--warmup_steps", type=int, default=10)
    arg_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Scale up the batch size")
    arg_parser.add_argument("--weight_decay", type=float, default=0.0001)
    
    # Early Stopping parameters
    arg_parser.add_argument(
        "--early_stopping",
        type=str2bool,
        nargs='?',
        const=True,
        default=True
    )
    arg_parser.add_argument("--patience", type=int, default=10)
    arg_parser.add_argument("--criteria", type=str, default='loss')
    arg_parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    arg_parser.add_argument("--resume_from_epoch", type=int, default=0)

    # Privacy parameters
    arg_parser.add_argument(
        "--private",
        type=str2bool,
        nargs='?',
        const=True,
        default=False
    )
    arg_parser.add_argument("--l2_norm_clip", type=float, default=1)
    arg_parser.add_argument("--noise_multiplier", type=float, default=0.81)
    arg_parser.add_argument(
        "--sampling_method",
        type=str,
        default="poisson_sampling",
        help="Sampling method for the privacy accountant, either 'poisson_sampling' or 'sampling_without_replacement'"
    )

    # Generation parameters
    arg_parser.add_argument(
        "--generate",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="if only generate, no training",
    )
    arg_parser.add_argument("--num_beams", type=int, default=3)
    arg_parser.add_argument(
        "--cross_generate",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Cross generate with other datasets",
    )

    args = arg_parser.parse_args()

    return args


class Settings:
    """
        Configuration for the project.
    """

    def __init__(self, args):
        # args, e.g., the output of settings.parse_arguments()
        self.args = args
