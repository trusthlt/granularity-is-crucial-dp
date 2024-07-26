from datasets import load_dataset_builder
from utils import compute_epsilons
from tqdm import tqdm
import numpy as np
import argparse
import time


def main():
    timeout = time.time() + 60 * 1

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--dataset", type=str, required=True)
    arg_parser.add_argument("--lang_pair", type=str, default='de-en')
    arg_parser.add_argument("--batch_size", type=int, required=True)
    arg_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Scale up the batch size")
    arg_parser.add_argument("--device_count", type=int, required=True)
    arg_parser.add_argument("--epochs", type=int, required=True)
    arg_parser.add_argument(
        "--sampling_method",
        type=str,
        required=True,
        help="Sampling method for the privacy accountant, either 'poisson_sampling' or 'sampling_without_replacement'"
    )
    arg_parser.add_argument("--epsilon", type=float, required=True)
    arg_parser.add_argument("--delta", type=float, default='1e-8')


    args = arg_parser.parse_args()

    # Set values
    total_batch_size = args.batch_size * args.device_count
    epochs = args.epochs
    target_ep = args.epsilon
    ds_builder = load_dataset_builder(args.dataset, args.lang_pair)
    len_train_dataset = ds_builder.info.splits['train'].num_examples
    print(f"original len train: {len_train_dataset}")

    noise_multipliers = np.concatenate(
        (np.arange(0.0, 1.0, 0.01),
         np.arange(1.0, 5.0, 0.1),
         np.arange(5.0, 100, 0.5),
         np.array([128, 256])
         )
    )
    _, remainder = divmod(len_train_dataset, args.device_count)
    actual_compute_len_train = len_train_dataset if remainder == 0 else len_train_dataset + remainder

    low_bound = 0
    high_bound = 0
    next_stop = False
    epsilon_low_bound = 0
    for noise_multiplier in tqdm(noise_multipliers, desc="First search bound"):
        noise_multiplier = round(noise_multiplier, 2)
        if next_stop: break
        epsilon = compute_epsilons(
            actual_compute_len_train,
            total_batch_size * args.gradient_accumulation_steps,
            noise_multiplier,
            epochs,
            args.delta,
            sampling_method="poisson_sampling",
        )
        if epsilon < target_ep:
            high_bound = noise_multiplier
            next_stop = True
        else:
            low_bound = noise_multiplier
            epsilon_low_bound = epsilon

    low_bound_new = 0
    while epsilon_low_bound != target_ep:
        if time.time() > timeout or len(str(low_bound)) >= 20:
            epsilon_low_bound = compute_epsilons(
                actual_compute_len_train,
                total_batch_size * args.gradient_accumulation_steps,
                high_bound,
                epochs,
                args.delta,
                sampling_method="poisson_sampling",
            )
            low_bound_new = high_bound
            break
        num = ["0", "01", "02", "03", "04", "05", "06", "07", "08", "09", 1, 2, 3, 4, 5, 6, 7, 8, 9]
        next_stop = False
        for i in num:
            if next_stop: break
            noise_multiplier = float(str(low_bound) + str(i))
            epsilon = compute_epsilons(
                actual_compute_len_train,
                total_batch_size * args.gradient_accumulation_steps,
                noise_multiplier,
                epochs,
                args.delta,
                sampling_method="poisson_sampling",
            )
            if epsilon < target_ep:
                high_bound = noise_multiplier
                next_stop = True
            else:
                if i == "0" and str(high_bound)[-2] == "0":
                    noise_multiplier = float(str(high_bound)[:-1] + "0" + str(high_bound)[-1])
                    epsilon = compute_epsilons(
                        actual_compute_len_train,
                        total_batch_size * args.gradient_accumulation_steps,
                        noise_multiplier,
                        epochs,
                        args.delta,
                        sampling_method="poisson_sampling",
                    )
                    while epsilon < target_ep:
                        noise_multiplier = float(str(noise_multiplier)[:-1] + "0" + str(noise_multiplier)[-1])
                        epsilon = compute_epsilons(
                            actual_compute_len_train,
                            total_batch_size * args.gradient_accumulation_steps,
                            noise_multiplier,
                            epochs,
                            args.delta,
                            sampling_method="poisson_sampling",
                        )
                low_bound_new = noise_multiplier
                epsilon_low_bound = epsilon
                epsilon_low_bound = round(epsilon_low_bound, 10)
        low_bound = low_bound_new
        print("bound noise:", low_bound)
        print("bound epsilon:", epsilon_low_bound)

    print("actual_compute_len_train:", actual_compute_len_train)
    print("devices:", args.device_count)
    print("total_batch_size:", total_batch_size)
    print("gradient_accumulation_steps:", args.gradient_accumulation_steps)
    print("accumulation_batch_size:", total_batch_size * args.gradient_accumulation_steps)
    print("epochs:", epochs)
    print("sampling_method:", args.sampling_method)
    print("input noise_multiplier:", low_bound_new)
    print("Epsilon:", epsilon_low_bound)
    print("\n")


if __name__ == '__main__':
    main()
