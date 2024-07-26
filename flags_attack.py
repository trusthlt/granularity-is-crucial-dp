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

    arg_parser.add_argument("--dataset_infer", type=str, required=True)
    arg_parser.add_argument("--dataset_loss", type=str, required=True)
    arg_parser.add_argument(
        "--base_model",
        type=str,
        default='agemagician/mlong-t5-tglobal-base',
        help="Path to model"
    )
    arg_parser.add_argument(
        "--target_model",
        type=str,
        default='agemagician/mlong-t5-tglobal-base',
        help="Path to model"
    )
    arg_parser.add_argument("--lang_pair", type=str, default='de-en')
    arg_parser.add_argument("--seed", type=int, default=666)
    arg_parser.add_argument("--batch_size", type=int, default=8, help="batch size per device")
    arg_parser.add_argument("--input_max_seq_len", type=int, default=64)
    arg_parser.add_argument("--output_max_seq_len", type=int, default=512)

    args = arg_parser.parse_args()
    return args


class AttackSettings:
    """
        Configuration for the project.
    """

    def __init__(self, args):
        # args, e.g., the output of settings.parse_arguments()
        self.args = args