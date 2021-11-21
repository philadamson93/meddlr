from numbers import Number
from typing import Dict, Sequence, Union

import meddlr.ops.complex as cplx
from meddlr.data.transforms.transform import Normalizer
from meddlr.evaluation.testing import flatten_results_dict
from meddlr.forward import SenseModel
from meddlr.transforms.build import (
    build_iter_func,
    build_scheduler,
    build_transforms,
    seed_tfm_gens,
)
from meddlr.transforms.gen import RandomTransformChoice
from meddlr.transforms.mixins import DeviceMixin, GeometricMixin
from meddlr.transforms.tf_scheduler import SchedulableMixin, TFScheduler
from meddlr.transforms.transform import NoOpTransform, Transform, TransformList
from meddlr.transforms.transform_gen import TransformGen
from meddlr.utils import env


class MRIReconAugmentor(DeviceMixin):
    """
    The class that manages the organization, generation, and application
    of deterministic and random transforms for MRI reconstruction.
    """

    def __init__(
        self,
        tfms_or_gens: Sequence[Union[Transform, TransformGen]],
        aug_sensitivity_maps: bool = True,
        seed: int = None,
        device=None,
    ) -> None:
        """
        Args:
            aug_sensitivity_maps (bool, optional): If ``True``, apply equivariant,
                image-based transforms to the sensivitiy map.
        """
        if isinstance(tfms_or_gens, TransformList):
            tfms_or_gens = tfms_or_gens.transforms
        self.tfms_or_gens = tfms_or_gens
        self.aug_sensitivity_maps = aug_sensitivity_maps

        if device is not None:
            self.to(device)
        if seed is None:
            seed_tfm_gens(self.tfms_or_gens, seed=seed)

    def __call__(
        self,
        kspace,
        maps=None,
        target=None,
        normalizer: Normalizer = None,
        mask=None,
        mask_gen=None,
        skip_tfm=False,
    ):
        if skip_tfm:
            tfms_equivariant, tfms_invariant = [], []
        else:
            # For now, we assume that transforms generated by RandomTransformChoice
            # return random transforms of the same type (equivariant or invariant).
            # We don't have to filter these transforms as a result.
            transform_gens = [
                x.get_transform() if isinstance(x, RandomTransformChoice) else x
                for x in self.tfms_or_gens
            ]
            if any(isinstance(x, (list, tuple)) for x in transform_gens):
                _temp = []
                for t in transform_gens:
                    if isinstance(t, (list, tuple)):
                        _temp.extend(t)
                    else:
                        _temp.append(t)
                transform_gens = _temp

            tfms_equivariant, tfms_invariant = self._classify_transforms(transform_gens)

        use_img = normalizer is not None or len(tfms_equivariant) > 0

        # Apply equivariant transforms to the SENSE reconstructed image.
        # Note, RSS reconstruction is not currently supported.
        if use_img:
            if mask is True:
                mask = cplx.get_mask(kspace)
            A = SenseModel(maps, weights=mask)
            img = A(kspace, adjoint=True)

        if len(tfms_equivariant) > 0:
            img, target, maps = self._permute_data(img, target, maps, spatial_last=True)
            img, target, maps, tfms_equivariant = self._apply_te(
                tfms_equivariant, img, target, maps
            )
            img, target, maps = self._permute_data(img, target, maps, spatial_last=False)

        if len(tfms_equivariant) > 0:
            A = SenseModel(maps)
            kspace = A(img)

        if mask_gen is not None:
            kspace, mask = mask_gen(kspace)
            img = SenseModel(maps, weights=mask)(kspace, adjoint=True)

        if normalizer:
            normalized = normalizer.normalize(
                **{"masked_kspace": kspace, "image": img, "target": target, "mask": mask}
            )
            kspace = normalized["masked_kspace"]
            target = normalized["target"]
            mean = normalized["mean"]
            std = normalized["std"]
        else:
            mean, std = None, None

        # Apply invariant transforms.
        if len(tfms_invariant) > 0:
            kspace = self._permute_data(kspace, spatial_last=True)
            kspace, tfms_invariant = self._apply_ti(tfms_invariant, kspace)
            kspace = self._permute_data(kspace, spatial_last=False)

        out = {"kspace": kspace, "maps": maps, "target": target, "mean": mean, "std": std}

        for s in self.schedulers():
            s.step(kspace.shape[0])

        return out, tfms_equivariant, tfms_invariant

    def schedulers(self):
        schedulers = [
            tfm.schedulers() for tfm in self.tfms_or_gens if isinstance(tfm, SchedulableMixin)
        ]
        return [x for y in schedulers for x in y]

    def get_tfm_gen_params(self, scalars_only: bool = True):
        """Get dictionary of scheduler parameters."""
        schedulers: Dict[str, Sequence[TFScheduler]] = {
            type(tfm).__name__: tfm._get_param_values(use_schedulers=True)
            for tfm in self.tfms_or_gens
            if isinstance(tfm, SchedulableMixin)
        }
        params = {}
        for tfm_name, p in schedulers.items():
            p = flatten_results_dict(p, delimiter=".")
            # Filter out values that are not scalars
            p = {f"{tfm_name}/{k}": v for k, v in p.items()}
            if scalars_only:
                p = {k: v for k, v in p.items() if isinstance(v, Number)}
            params.update(p)
        return params

    def _classify_transforms(self, transform_gens):
        tfms_equivariant = []
        tfms_invariant = []
        for tfm in transform_gens:
            if isinstance(tfm, TransformGen):
                tfm_kind = tfm._base_transform
            else:
                tfm_kind = type(tfm)
            assert issubclass(tfm_kind, Transform)

            if issubclass(tfm_kind, GeometricMixin):
                tfms_equivariant.append(tfm)
            else:
                tfms_invariant.append(tfm)
        return tfms_equivariant, tfms_invariant

    def _permute_data(self, *args, spatial_last: bool = False):
        out = []
        if spatial_last:
            for x in args:
                dims = (0,) + tuple(range(3, x.ndim)) + (1, 2)
                out.append(x.permute(dims))
        else:
            for x in args:
                dims = (0, x.ndim - 2, x.ndim - 1) + tuple(range(1, x.ndim - 2))
                out.append(x.permute(dims))
        return out[0] if len(out) == 1 else tuple(out)

    def _apply_te(self, tfms_equivariant, image, target, maps):
        """Apply equivariant transforms.

        These transforms affect both the input and the target.
        """
        tfms = []
        for g in tfms_equivariant:
            tfm: Transform = g.get_transform(image) if isinstance(g, TransformGen) else g
            if isinstance(tfm, NoOpTransform):
                continue
            image = tfm.apply_image(image)
            if target is not None:
                target = tfm.apply_image(target)
            if maps is not None and self.aug_sensitivity_maps:
                maps = tfm.apply_maps(maps)
            tfms.append(tfm)
        return image, target, maps, TransformList(tfms, ignore_no_op=True)

    def _apply_ti(self, tfms_invariant, kspace):
        """Apply invariant transforms.

        These transforms affect only the input, not the target.
        """
        tfms = []
        for g in tfms_invariant:
            tfm: Transform = g.get_transform(kspace) if isinstance(g, TransformGen) else g
            if isinstance(tfm, NoOpTransform):
                continue
            kspace = tfm.apply_kspace(kspace)
            tfms.append(tfm)
        return kspace, TransformList(tfms, ignore_no_op=True)

    def reset(self):
        for g in self.tfms_or_gens:
            if isinstance(g, TransformGen):
                g.reset()

    def to(self, device):
        tfms = [tfm for tfm in self.tfms_or_gens if isinstance(tfm, DeviceMixin)]
        for t in tfms:
            t.to(device)
        return self

    @classmethod
    def from_cfg(cls, cfg, aug_kind, seed=None, device=None, **kwargs):
        mri_tfm_cfg = None
        assert aug_kind in ("aug_train", "consistency")
        if aug_kind == "aug_train":
            mri_tfm_cfg = cfg.AUG_TRAIN.MRI_RECON
        elif aug_kind == "consistency":
            mri_tfm_cfg = cfg.MODEL.CONSISTENCY.AUG.MRI_RECON

        if seed is None and env.is_repro():
            seed = cfg.SEED

        tfms_or_gens = build_transforms(cfg, mri_tfm_cfg.TRANSFORMS, seed=seed, **kwargs)
        scheduler_p = dict(mri_tfm_cfg.SCHEDULER_P)
        ignore_scheduler = scheduler_p.pop("IGNORE", False)
        if len(scheduler_p) and not ignore_scheduler:
            scheduler_p["params"] = ["p"]
            tfms = [tfm for tfm in tfms_or_gens if isinstance(tfm, TransformGen)]
            for tfm in tfms:
                scheduler = build_scheduler(cfg, scheduler_p, tfm)
                tfm.register_schedulers([scheduler])

        if aug_kind in ("aug_train",) and cfg.DATALOADER.NUM_WORKERS > 0:
            func = build_iter_func(cfg.SOLVER.TRAIN_BATCH_SIZE, cfg.DATALOADER.NUM_WORKERS)
            tfms = [tfm for tfm in tfms_or_gens if isinstance(tfm, SchedulableMixin)]
            for tfm in tfms:
                for s in tfm._schedulers:
                    s._iter_fn = func

        return cls(
            tfms_or_gens,
            aug_sensitivity_maps=mri_tfm_cfg.AUG_SENSITIVITY_MAPS,
            seed=seed,
            device=device,
        )
