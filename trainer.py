from models.ModelWithLoss import ModelandLossBase
import torch.optim
from torch.utils.data import DataLoader, Subset, random_split, RandomSampler
from tensorboardX import SummaryWriter
from datetime import datetime
import os


def train(modelloss: ModelandLossBase, dataset: DataSetBase, cfg: dict):
    # ------------------------------
    # figuring out configuration
    # -----------------------------
    # learning group <<<<<<<
    num_epoch = cfg["num_epoch"]
    batch_sz = cfg["batch_size"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    # log group <<<<<<<<
    save_epoch = cfg["save_epoch"]
    log_prefix = cfg["log_prefix"]
    save_path = log_prefix + cfg["model_name"] + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        check_point = torch.load(cfg["check_point"])
    except FileNotFoundError:
        print(f"cannot open check point file {cfg['check_point']}")
        check_point = None
    # check point group <<<<<<<<<
    begin_epoch = -1
    model = modelloss.model
    if check_point is not None:
        model.load_state_dict(check_point["state_dict"])
        begin_epoch = check_point["epoch"]
        print(f"load state dict of epoch:{begin_epoch}-1")
    else:
        check_point = {}
        modelloss.initial_weights()
        print(f"initial the model weights")

    # TODO: learning rate scheduler
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    if "optimizer" in check_point.keys():
        check_point["optimizer"]["param_groups"][0]["lr"] = lr
        check_point["optimizer"]["param_groups"][0]["weight_decay"] = weight_decay
        optimizer.load_state_dict(check_point["optimizer"])

    # validation group <<<<<<<<<<<
    perform_validation = dataset.haveGT  # if not groud truth, cannot perform validation
    perform_validation2 = isinstance(dataset, DAVISSeq)
    valid_visual_out = cfg["write_validate_result"]
    validate_gtnum = cfg["validate_num"] if 0 < cfg["validate_num"] < len(dataset.eval()) else len(dataset.eval())

    datasubset = Subset(dataset.eval(), torch.randperm(len(dataset.eval()))[:validate_gtnum])
    evaldata = DataLoader(datasubset, batch_size=batch_sz, shuffle=False, pin_memory=True, drop_last=True,)

    # framenum = dataset.framenum
    # selected_idx = torch.linspace(0, framenum - 2, validate_gtnum).type(torch.int).tolist()
    # val_frames = [(dataset.get_tensorframe(_idx), dataset.get_tensorframe(_idx+1)) for _idx in selected_idx]
    # val_gtflows = [dataset.get_gtflow(_idx) for _idx in selected_idx]
    # val_gtocc = [dataset.get_gtocc(_idx) for _idx in selected_idx]
    validate_freq = cfg["valid_freq"]

    # training group <<<<<<<<<<<<<<
    train_report_freq = cfg["train_report_freq"]
    trainingdata = DataLoader(dataset.train(), batch_sz, pin_memory=True, drop_last=True,
                              sampler=RandomSampler(dataset.train(), True, cfg["sample_num_per_epoch"]))

    cfg_run_dir = cfg["run_dir"]
    log_dir = log_prefix + cfg_run_dir + datetime.now().strftime("%Y%m%d_%H%M")
    tensorboard = SummaryWriter(log_dir)
    # write configure of the current training
    cfg_out = log_prefix + cfg_run_dir + "configs.txt"
    with open(cfg_out, 'a') as cfg_outfile:
        loss_str = ', '.join([f'{k}:{v}' for k, v in cfg['loss_weights'].items()])
        occ_str = ', '.join([f'{k}:{v}' for k, v in cfg['occ_cfg'].items()]) if "occ_cfg" in cfg.keys() else "None"
        cfg_str = f"\n-------------------------------------------------\n" \
                  f"|Name: {log_dir}|\n" \
                  f"-----------------\n" \
                  f"Comment: {cfg['comment']}\n" \
                  f"Dataset: {dataset.name}, len={len(dataset)}\n" \
                  f"Model: {cfg['model_name']}\n" \
                  f"Loss: {loss_str}\n" \
                  f"Occ_cfg: {occ_str}\n" \
                  f"Training: checkpoint={begin_epoch}, bsz={batch_sz}, lr={lr}, weightdecay={weight_decay}\n" \
                  f"    saved_epoch={save_epoch}\n" \
                  f"Validation: {perform_validation}, gtnum={validate_gtnum}, freq={validate_freq}\n" \
                  f"\n"
        cfg_outfile.write(cfg_str)

    # =============================================
    # kick-off training
    # =============================================
    dataset.train()
    for epoch in range(begin_epoch, num_epoch):
        # epoch_mean_loss, epoch_mean_lossn, epoch_loss_dict = 0, 0, {}
        for iternum, datas in enumerate(trainingdata):

            # one epoch
            loss, loss_dict = modelloss(*datas)

            # recored iter loss
            step = len(trainingdata) * epoch + iternum
            tensorboard.add_scalar("final_loss", float(loss), step)
            for lossname, lossval in loss_dict.items():
                tensorboard.add_scalar(lossname, lossval, step)
            # record epoch_mean_loss
            # epoch_mean_loss += loss
            # epoch_mean_lossn += 1
            # epoch_loss_dict = {k: (epoch_loss_dict[k] + loss_dict[k] if k in epoch_loss_dict else 0)
            #                    for k in loss_dict.keys()}
            # output loss message
            if iternum % train_report_freq == 0:
                # output iter infomation
                loss_str = " | ".join([f"{k}: {v:.3f}" for k, v in loss_dict.items()])

                curlr = optimizer.state_dict()["param_groups"][0]["lr"]
                print(f"    iter {iternum}/{len(trainingdata)}: lr:{curlr} loss: {loss:.3f} | {loss_str}", flush=True)

            # perform evaluation
            if perform_validation and iternum % validate_freq == 0:
                print("  eval...", end="")
                dataset.eval()
                val_dict, val_display = {}, None
                for val_frame0s, val_frame1s, val_gtflows, val_gtocc in evaldata:
                    _val_dict, val_display = modelloss.valid_forward(val_frame0s, val_frame1s,
                                                                     val_gtflows, val_gtocc, not bool(val_dict))

                    val_dict = {k: val_dict[k] + v if k in val_dict.keys() else v
                                for k, v in _val_dict.items()}

                    if val_display is not None and valid_visual_out:
                        tensorboard.add_image("valid_Flow", val_display, step)

                val_dict = {k: v / len(evaldata) for k, v in val_dict.items()}
                # collect iou
                if "val_OCCiou_i" in val_dict.keys():
                    val_dict["val_OCCiou"] = val_dict["val_OCCiou_i"] / val_dict["val_OCCiou_u"]
                    val_dict.pop("val_OCCiou_i", None)
                    val_dict.pop("val_OCCiou_u", None)
                for key, value in val_dict.items():
                    tensorboard.add_scalar(key, value, step)

                print(" eval:" + ' | '.join([f"{k}: {v:.3f}" for k, v in val_dict.items()]), flush=True)
                dataset.train()

            if perform_validation2 and iternum % validate_freq == 0:
                evaluator1 = LongRangeSegIou(10)
                evaluator2 = FeaturePointsDrift(15, 2)
                val_dict, val_display = evaluator1.evaluate(modelloss, dataset)
                if val_display is not None and valid_visual_out:
                    tensorboard.add_image("valid_Seg", val_display, step)
                for key, value in val_dict.items():
                    tensorboard.add_scalar(key, value, step)
                print(" eval:" + ' | '.join([f"{k}: {v:.3f}" for k, v in val_dict.items()]), flush=True)
                dataset.train()
                val_dict, _ = evaluator2.evaluate(modelloss, dataset)
                for key, value in val_dict.items():
                    tensorboard.add_scalar(key, value, step)
                print(" eval:" + ' | '.join([f"{k}: {v:.3f}" for k, v in val_dict.items()]), flush=True)
                dataset.train()

            # first evaluate and then backward so that first evaluate is always the same
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # output epoch info
        # epoch_mean_loss /= epoch_mean_lossn
        # loss_str = " | ".join([f"{k}:{v:.3f}".format(k, v / epoch_mean_lossn) for k, v in epoch_loss_dict.items()])
        # print("epoch {}: loss: {:.3f} | {}".format(epoch, epoch_mean_loss,  loss_str))
        print(f"epoch {epoch} ================================")
        if epoch in save_epoch:
            torch.save({
                "cfg": cfg_str,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, f"{save_path}{epoch}.ckpt")
            print(f"checkpoint saved {epoch}.ckpt, will start from {epoch + 1} when loading")

    tensorboard.close()
