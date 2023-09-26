"""Test the ssfd loss with motion."""
import os

import h5py
import numpy as np
import pandas as pd
import torch

from meddlr.config import get_cfg
from meddlr.data.build import build_recon_val_loader
from meddlr.engine import default_argument_parser, default_setup
from meddlr.evaluation import inference_on_dataset
from meddlr.evaluation.recon_evaluation import ReconEvaluator
from meddlr.utils.general import move_to_device
from meddlr.utils.logger import setup_logger

_FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
logger = None  # initialize in setup()


class LoadReconTransform:
    """A dummy data transform that loads a previously saved reconstruction for evaluation.
    Used for the SSFD reader study where original reconstructions were done outside
    of meddlr."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(
        self,
        kspace: torch.Tensor,
        maps: torch.Tensor,
        target: torch.Tensor,
        fname: str,
        slice_id: int,
        is_unsupervised: bool = False,
        fixed_acc: bool = False,
    ):
        kspace = torch.as_tensor(kspace)
        maps = torch.as_tensor(maps)
        target = torch.as_tensor(target)

        out = {"kspace": kspace, "maps": maps, "target": target}

        out["mean"] = torch.as_tensor([0.0])
        out["std"] = torch.as_tensor([1.0])

        # Load the saved reconstruction.
        recon_path = os.path.join(self.cfg.TEST.SAVED_RECON_DIR, fname.split(".")[0] + ".npy")
        if os.path.exists(recon_path):
            recon = np.load(recon_path)

        else:
            recon_path = os.path.join(self.cfg.TEST.SAVED_RECON_DIR, fname.split(".")[0] + ".h5")
            with h5py.File(recon_path, "r") as f:
                dset = f["pred"]
                recon = np.array(dset)

        recon_slice = recon[slice_id, :, :, :]
        out["pred"] = torch.as_tensor(recon_slice)

        return out


class NoOpModel(torch.nn.Module):
    def forward(self, inputs):
        inputs = move_to_device(inputs, cfg.MODEL.DEVICE)
        return {"pred": inputs.pop("pred"), "target": inputs.pop("target")}


def build_data_loader(cfg, dataset_name: str, transform: LoadReconTransform):
    dl = build_recon_val_loader(
        cfg=cfg, dataset_name=dataset_name, as_test=True, add_noise=False, add_motion=False
    )
    dl.dataset.transform = transform
    return dl


def setup(args):
    """
    Create configs and perform basic setups.
    We do not save the config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if opts and opts[0] == "--":
        opts = opts[1:]
    cfg.merge_from_list(opts)
    cfg.freeze()
    default_setup(cfg, args, save_cfg=False)

    # Setup logger for test results
    global logger
    logger = setup_logger(os.path.join(cfg.OUTPUT_DIR, args.save_dir), name=_FILE_NAME)

    return cfg


def default_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--save-dir", type=str, default="test_results", help="Directory to save test results."
    )
    parser.add_argument(
        "--saved-recon-dir",
        type=str,
        default="",
        help="Directory to load the image reconstructions from.",
    )
    return parser


def run_eval(cfg, dataset: str):
    model = NoOpModel()

    logger.info("==" * 40)
    logger.info("Evaluating {} ...".format(dataset))
    logger.info("==" * 40)

    save_dir = os.path.join(cfg.OUTPUT_DIR, args.save_dir, dataset)

    evaluator = ReconEvaluator(
        dataset_name=dataset,
        cfg=cfg,
        device=cfg.MODEL.DEVICE,
        output_dir=save_dir,
        save_scans=False,
        metrics=cfg.TEST.VAL_METRICS.RECON,
        layer_names=cfg.TEST.VAL_METRICS.LAYER_NAMES,
        flush_period=cfg.TEST.FLUSH_PERIOD,
        prefix="test",
        group_by_scan=False,
    )

    transform = LoadReconTransform(cfg)
    dl = build_data_loader(cfg, dataset, transform)

    results = inference_on_dataset(model, data_loader=dl, evaluator=evaluator)
    results = pd.DataFrame(results).T.reset_index().rename(columns={"index": "scan_name"})
    return results


if __name__ == "__main__":
    args = default_parser().parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    datasets = cfg.DATASETS.TEST

    results = []
    for dataset in datasets:
        results.append(run_eval(cfg.clone(), dataset))
    results = pd.concat(results)
    results.to_csv(os.path.join(cfg.OUTPUT_DIR, args.save_dir, "test_results.csv"), index=False)
