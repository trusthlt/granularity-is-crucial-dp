from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data import default_collate
from opacus.data_loader import logger
import torch


def wrap_collate_with_empty(collate_fn):
    """
    Wraps given collate function to handle empty batches.
    Args:
        collate_fn: collate function to wrap
    Returns:
        New collate function, which is equivalent to input ``collate_fn`` for non-empty
        batches and outputs dictionary with ``skip_batch`` as the only key if
        the input batch is of size 0
    """

    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return {'skip_batch': True}

    return collate


class CustomDPDataLoader(DataLoader):
    """
    Custom DPDataLoader that not requires splitting batches if larger than the maximum batch size.
    Forked from the original DPDataLoader in opacus.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_rate: float,
        collate_fn: None,
        drop_last: bool = False,
        generator=None,
        distributed: bool = False,
        **kwargs,
    ):

        self.sample_rate = sample_rate
        self.distributed = distributed

        batch_sampler = CustomDPSampler(
            num_samples=len(dataset),  # type: ignore[assignment, arg-type]
            sample_rate=sample_rate,
            generator=generator,
        )
        if collate_fn is None:
            collate_fn = default_collate

        if drop_last:
            logger.warning(
                "Ignoring drop_last as it is not compatible with DPDataLoader."
            )

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=wrap_collate_with_empty(
                collate_fn=collate_fn,
            ),
            generator=generator,
            **kwargs,
        )

    @classmethod
    def from_data_loader(
            cls, data_loader: DataLoader, *, distributed: bool = False, generator=None, model_name=None,
    ):

        if isinstance(data_loader.dataset, IterableDataset):
            raise ValueError("Uniform sampling is not supported for IterableDataset")

        return cls(
            dataset=data_loader.dataset,
            sample_rate=1 / len(data_loader),
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=generator if generator else data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
            distributed=distributed,
        )


class CustomDPSampler(UniformWithReplacementSampler):
    def __init__(self, *, num_samples: int, sample_rate: float, generator=None, steps=None):
        """
        Args:
            num_samples: number of samples to draw.
            sample_rate: probability used in sampling.
            generator: Generator used in sampling.
            steps: Number of steps (iterations of the Sampler)
            model_name: Name of the model to be used for the batch size
        """
        super().__init__(num_samples=num_samples, sample_rate=sample_rate, generator=generator)
        if steps is not None:
            self.steps = steps
        else:
            self.steps = int(1 / self.sample_rate)

    def __iter__(self):
        num_batches = self.steps
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            yield indices

            num_batches -= 1
