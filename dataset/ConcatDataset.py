from torch.utils.data import Dataset
import numpy as np
import bisect


class ConcatDatasetMy(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l_ = e
            r.append(l_ + s)
            s += l_
        return r

    def __init__(self, datasets, frequency):
        super(ConcatDatasetMy, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        assert len(datasets) == len(frequency), "frequency of each datasets should be same as datasets"
        self.datasets = list(datasets)
        self.original_size = [len(l_) for l_ in self.datasets]
        self.used_size = [int(len(l_) * freq) for l_, freq in zip(datasets, frequency)]
        self.cumulative_sizes = self.cumsum(self.used_size)
        self.samplemap = [
            np.random.choice(allsz, sz, replace=(allsz < sz))
            for allsz, sz in zip(self.original_size, self.used_size)
        ]
        self.used_count = [0] * len(self.used_size)
        self._cur_dataset_idx = 0

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        self._cur_dataset_idx = dataset_idx
        # update samplemap
        self.used_count[dataset_idx] += 1
        self.name = self.datasets[self._cur_dataset_idx].name
        if self.used_count[dataset_idx] > self.used_size[dataset_idx]:
            self.samplemap[dataset_idx] = np.random.choice(
                self.original_size[dataset_idx],
                self.used_size[dataset_idx],
                replace=(self.original_size[dataset_idx] < self.used_size[dataset_idx])
            )
            self.used_count[dataset_idx] = 0
        return self.datasets[dataset_idx][self.samplemap[dataset_idx][sample_idx]]

    @property
    def _cur_file_base(self):
        return self.datasets[self._cur_dataset_idx]._cur_file_base
