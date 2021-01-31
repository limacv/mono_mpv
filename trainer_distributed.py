import torch.optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset, random_split, RandomSampler

from datetime import datetime
import time
from utils import *
from util.config import fakeSummaryWriter
from tensorboardX import SummaryWriter

plane_num = 24


def train(cfg: dict):
    model = select_module(cfg["model_name"])
    modelloss = select_modelloss(cfg["modelloss_name"])(model, cfg)

    num_epoch = cfg["num_epoch"]
    batch_sz = cfg["batch_size"]
    lr = cfg["lr"]
    world_size = cfg["world_size"]
    local_rank = cfg["local_rank"]

    # figuring dispout all the path
    log_prefix = cfg["log_prefix"]
    if "id" in cfg.keys():
        unique_id = f"{cfg['id']}_{datetime.now().strftime('%d%H%M')}"
    else:
        unique_id = datetime.now().strftime("%m%d_%H%M%S")
    mpi_save_path = os.path.join(log_prefix, cfg["mpi_outdir"], f"{unique_id}")
    checkpoint_path = os.path.join(log_prefix, cfg["checkpoint_dir"])
    tensorboard_log_prefix = os.path.join(log_prefix, cfg["tensorboard_logdir"])
    cfgstr_out = os.path.join(tensorboard_log_prefix, "configs.txt")
    log_dir = os.path.join(tensorboard_log_prefix, str(unique_id))

    mkdir_ifnotexist(mpi_save_path)
    mkdir_ifnotexist(checkpoint_path)
    mkdir_ifnotexist(tensorboard_log_prefix)

    begin_epoch = smart_load_checkpoint(checkpoint_path, cfg, model)

    savepth_iter_freq = cfg["savepth_iter_freq"]
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # evalset = RealEstate10K_Seq(is_train=False, seq_len=cfg["evalset_seq_length"])
    # evalset = StereoBlur_Img(is_train=False)
    evalset = select_dataset(cfg["evalset"], False, cfg)
    validate_gtnum = cfg["validate_num"] if 0 < cfg["validate_num"] < len(evalset) else len(evalset)

    datasubset = Subset(evalset, torch.randperm(len(evalset))[:validate_gtnum])
    evaluatedata = DataLoader(datasubset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True,
                              )
    validate_freq = cfg["valid_freq"]

    # trainset = RealEstate10K_Seq(is_train=True, seq_len=cfg["dataset_seq_length"])
    trainset = select_dataset(cfg["trainset"], True, cfg)
    train_report_freq = cfg["train_report_freq"]
    distributedSampler = DistributedSampler(trainset, num_replicas=world_size, rank=local_rank)
    trainingdata = DataLoader(trainset, batch_sz,
                              pin_memory=True, drop_last=True,
                              sampler=distributedSampler)

    loss_str = ', '.join([f'{k}:{v}' for k, v in cfg['loss_weights'].items()])
    cfg_str = f"\n-------------------------------------------------\n" \
              f"|Name: {log_dir}|\n" \
              f"-----------------\n" \
              f"Comment: {cfg['comment']}\n" \
              f"Dataset: train:{trainset.name}, len={len(trainset)}, eval:{evalset.name}, len={len(evalset)}\n" \
              f"Model: {cfg['model_name']}, ModelLoss: {cfg['modelloss_name']}\n" \
              f"Loss: {loss_str}\n" \
              f"Training: checkpoint={cfg['check_point']}, bsz={batch_sz}, lr={lr}\n" \
              f"Validation: gtnum={validate_gtnum}, freq={validate_freq}\n" \
              f"\n"

    if local_rank == 0:
        tensorboard = SummaryWriter(log_dir)
        # write configure of the current training
        with open(cfgstr_out, 'a') as cfg_outfile:
            cfg_outfile.write(cfg_str)
        print(cfg_str, flush=True)
    else:
        tensorboard = fakeSummaryWriter()

    # =============================================
    # kick-off training
    # =============================================
    modelloss = DistributedDataParallel(modelloss, device_ids=[local_rank, ], output_device=local_rank,
                                        find_unused_parameters=True)
    start_time = time.time()
    for epoch in range(begin_epoch, num_epoch):
        distributedSampler.set_epoch(epoch)
        for iternum, datas in enumerate(trainingdata):
            step = len(trainingdata) * epoch + iternum
            # one epoch
            loss_dict = modelloss(*datas, step=step)
            loss = loss_dict["loss"]
            loss_dict = loss_dict["loss_dict"]
            loss = loss.mean()
            loss_dict = {k: v.mean() for k, v in loss_dict.items()}

            # debug usage
            if loss_dict["scale"] > 1000:
                print(f"RANK_{local_rank}:: detect large scale, "
                      f"current base: {trainset.name}: {trainset._cur_file_base}")
            elif loss_dict["scale"] < 1/1000:
                print(f"RANK_{local_rank}:: detect small scale, "
                      f"current base: {trainset.name}: {trainset._cur_file_base}")
            # first evaluate and then backward so that first evaluate is always the same
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # recored iter loss
            tensorboard.add_scalar("final_loss", float(loss), step)
            for lossname, lossval in loss_dict.items():
                tensorboard.add_scalar(lossname, lossval, step)

            # output loss message
            if step % train_report_freq == 0:
                time_per_iter = (time.time() - start_time) / train_report_freq
                # output iter infomation
                loss_str = " | ".join([f"{k}:{v:.3f}" for k, v in loss_dict.items()])
                print(f"    Local Rank: {local_rank}:")
                print(f"    iter {iternum}/{len(trainingdata)}::loss:{loss:.3f} | {loss_str} "
                      f"| time:{time_per_iter:.1f}s", flush=True)
                start_time = time.time()

            # perform evaluation
            if step % validate_freq == 0:
                val_dict, val_display = {}, None
                vis_id = np.random.randint(0, len(evaluatedata))
                for i, evaldatas in enumerate(evaluatedata):
                    _val_dict = modelloss.module.valid_forward(*evaldatas, visualize=(i == vis_id))

                    if i == vis_id:
                        for k in _val_dict.copy().keys():
                            if "vis_" in k:
                                tensorboard.add_image(k, _val_dict[k], step, dataformats='HWC')
                                _val_dict.pop(k, None)
                            if "save_" in k:
                                save_mpi(_val_dict.pop(k), mpi_save_path)

                    val_dict = {k: val_dict[k] + v if k in val_dict.keys() else v
                                for k, v in _val_dict.items()}

                val_dict = {k: v / len(evaluatedata) for k, v in val_dict.items()}
                for val_name, val_value in val_dict.items():
                    tensorboard.add_scalar(val_name, val_value, step)
                print(f"EVAL{local_rank}:: " + ' | '.join([f"{k}: {v:.3f}" for k, v in val_dict.items()]), flush=True)
                print(f"CHECKING Consistency:: {modelloss.module.parameters().__next__()[0, 0, 0]}")

            if (step + 1) % savepth_iter_freq == 0 and local_rank == 0:
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "cfg": cfg_str,
                    "state_dict": model.state_dict(),
                    # "optimizer": optimizer.state_dict()
                }, os.path.join(checkpoint_path, f"{unique_id}_r{local_rank}.pth"))
                print(f"checkpoint saved {epoch}{unique_id}_r{local_rank}.pth", flush=True)

    print("TRAINING Finished!!!!!!!!!!!!", flush=True)
    if tensorboard is not None:
        tensorboard.close()
