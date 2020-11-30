from typing import Union, Sequence, Dict


class Experiments:
    def __init__(self, _cfg: Dict, addfirst=True):
        self.cfg = _cfg
        self.newcfg_list = [{}, ] if addfirst else []
        self.acccfg = {}
        self.idx = 0
        self.default_device_id = 0

    def add_experiment(self, _newcfg: Dict):
        """
        :param _newcfg: the config(s) that need to modify
        The common workflow is first add_experiment to modify some cfg, then call add_experiments
        to add a series of experiments
        """
        self.newcfg_list.append(_newcfg)

    def add_experiments(self, key: Union[Sequence[str], str], values: Sequence):
        """
        :param key: if hierarcy keys, use list like ["occ_cfg", "estimate_occ"]
        :param values: a list of values to be estimated, like [0.2, 1., 5.]
        The common workflow is first add_experiment to modify some cfg, then call add_experiments
        to add a series of experiments
        """
        if isinstance(key, str):
            for value in values:
                self.newcfg_list.append({key: value})
        elif isinstance(key, Sequence):
            if len(key) > 2:
                raise NotImplementedError("Experiments: add_experiments with key length > 2 are not supported")
            newcfgs = [{key[-1]: v} for v in values]
            for k in key[:-1][::-1]:
                newcfgs = [{k: newcfg} for newcfg in newcfgs]
            self.newcfg_list += newcfgs

    def __len__(self):
        return len(self.newcfg_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self.newcfg_list):
            raise StopIteration
        self.cfg.update({"cuda_device": self.default_device_id})
        self.acccfg.update({"cuda_device": self.default_device_id})
        for k, v in self.newcfg_list[self.idx].items():
            if isinstance(v, dict):
                self.cfg[k].update(v)
                if k in self.acccfg.keys():
                    self.acccfg[k].update(v)
                else:
                    self.acccfg[k] = v
            else:
                self.cfg.update({k: v})
                self.acccfg.update({k: v})
        self.idx += 1

        # add device id
        num_gpu = self.cfg["gpu_num"] if "gpu_num" in self.cfg.keys() else 1
        self.cfg.update({"device_ids": list(range(self.default_device_id, self.default_device_id + num_gpu))})
        self.acccfg.update({"device_ids": list(range(self.default_device_id, self.default_device_id + num_gpu))})
        self.default_device_id += num_gpu
        return self.cfg

    def get_info_str(self):
        ret = '\n'.join([f"** {k}: {v}" for k, v in self.acccfg.items()])
        return ret


class fakeMultiProcessing:
    class Process:
        def __init__(self, target=None, args=None):
            pass

        def start(self):
            pass

        def join(self):
            pass

    @staticmethod
    def set_start_method(s):
        pass


class fakeSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def add_image(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    experiments = Experiments({})
    experiments.add_experiment({"check_point": 0, "occ_cfg": {"estimate_occ": False}})
    experiments.add_experiments(["loss_weights", "photo_loss"], [1, 2, 3, 4, 5])
    for newcfg in experiments:
        print(experiments.get_info_str())
        print("............")
        print(newcfg)
        print("///////////")
