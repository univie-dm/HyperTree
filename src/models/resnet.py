import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from src.models.abstract import AbstractModel, FeatureScaling
from src.models.utils import get_hyperbolic_encoding_layer_by_name, get_dtype_from_precision
from src.hyperbolic_math.src.manifolds import Euclidean

RESNET_MODELS = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
                 "resnet101": resnet101, "resnet152": resnet152}
RESNET_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}
# Views of an image that the pretraining consumes: its two augmented views
PRETRAINING_VIEWS = ("aug_1", "aug_2")
# Where the pretraining contrastive loss is computed: cosine similarity of the projection-head
# features (before the hyperbolic layer) or negative geodesic distance of the manifold embeddings
CONTRASTIVE_SPACES = ("euclidean", "hyperbolic")
# BioCLIP (Stevens et al., CVPR 2024): CLIP ViT-B/16 on TreeOfLife-10M, loaded
# via open_clip. 512-d image features. Always used as a frozen backbone.
BIOCLIP_DIM = 512
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _BioCLIPBackbone(nn.Module):
    """Frozen BioCLIP image encoder. The datamodule feeds ImageNet-normalized
    tensors; BioCLIP expects CLIP normalization, so we re-normalize on the fly
    (invert ImageNet stats, apply CLIP stats) as a fixed per-channel affine.
    Returns a 512-d image-feature vector per image."""

    def __init__(self):
        super().__init__()
        import open_clip  # lazy: only required when this backbone is used
        clip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        self.clip = clip_model
        in_mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        in_std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        clip_mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        clip_std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        # x_clip = (x_in * in_std + in_mean - clip_mean) / clip_std
        self.register_buffer("renorm_scale", in_std / clip_std)
        self.register_buffer("renorm_shift", (in_mean - clip_mean) / clip_std)

    def forward(self, x):
        x = x * self.renorm_scale + self.renorm_shift
        return self.clip.encode_image(x)


class HyperbolicResNet(AbstractModel):
    def __init__(
        self,
        # Architecture
        model_type: str | None = None,
        params_dtype: str = "float32",
        activation_function: str = "swish",
        feature_scaling: bool = True,
        hyperbolic_encoding: str | None = "HyperbolicLinearPP",
        latent_dim: int = 32,
        manifold_name: str = "PoincareBall",
        manifold_precision: str | torch.dtype = "float64",
        curvature_magnitude: float | torch.Tensor = 1.0,
        # Optimizer
        optimizer_name: str = "Adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_params: dict | None = None,
        # Evaluation & Logging
        stage: str = "pretrain",
        global_epoch: int = 0,
        log_interval: int = 10,
        evaluation_metrics: dict[str, bool] = {},
        # ResNet specific parameters
        projection_layer_dims: list[int] = [512, 256, 128],
        input_size: int = 32,
        contrastive_temperature: float = 0.1,
        contrastive_space: str = "euclidean",
        # External backbone checkpoint (e.g. DINO ResNet50 from Facebook)
        pretrained_weights_path: str | None = None,
    ):
        super().__init__(
            # Architecture
            model_type=model_type,
            params_dtype=params_dtype,
            activation_function=activation_function,
            feature_scaling=feature_scaling,
            hyperbolic_encoding=hyperbolic_encoding,
            latent_dim=latent_dim,
            manifold_name=manifold_name,
            manifold_precision=manifold_precision,
            curvature_magnitude=curvature_magnitude,
            # Optimizer
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_params=lr_scheduler_params,
            # Evaluation & Logging
            stage=stage,
            global_epoch=global_epoch,
            log_interval=log_interval,
            evaluation_metrics=evaluation_metrics,
        )
        self.save_hyperparameters()
        if contrastive_space not in CONTRASTIVE_SPACES:
            raise ValueError(f"contrastive_space must be one of {CONTRASTIVE_SPACES}, got '{contrastive_space}'")

        # Encoder — ResNet (torchvision) or frozen BioCLIP image encoder (open_clip)
        mt = self.hparams.model_type
        if mt in RESNET_MODELS:
            self.encoder = RESNET_MODELS[mt](weights=None)
            backbone_dim = RESNET_DIMS[mt]
            # Replace classification head with identity to expose pooled features.
            self.encoder.fc = nn.Identity()

            # Modify encoder for CIFAR-100 (32x32 images)
            if self.hparams.input_size == 32:
                print("Modifying ResNet encoder for CIFAR-100 (32x32 images).", flush=True)
                # Overwrite the first convolutional layer and maxpooling layer
                self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.encoder.maxpool = nn.Identity()  # type: ignore
            elif self.hparams.input_size == 64:
                print("Modifying ResNet encoder for 64x64 images (removing maxpool).", flush=True)
                self.encoder.maxpool = nn.Identity()  # type: ignore
        elif mt == "bioclip":
            print("Loading BioCLIP (hf-hub:imageomics/bioclip) via open_clip (frozen)", flush=True)
            self.encoder = _BioCLIPBackbone()
            backbone_dim = BIOCLIP_DIM
        else:
            raise ValueError(f"Invalid model type: {mt}")

        if self.hparams.pretrained_weights_path is not None:
            if mt == "bioclip":
                # BioCLIP already provides pretrained weights; refuse silent
                # overrides since the state-dict conventions differ.
                raise ValueError(
                    f"pretrained_weights_path is not supported for model_type={mt}; "
                    "pretrained weights are loaded via open_clip."
                )
            print(f"Loading backbone weights from {self.hparams.pretrained_weights_path}", flush=True)
            state = torch.load(self.hparams.pretrained_weights_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            print(f"Loaded external checkpoint: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        # Externally pretrained backbones are kept frozen
        if self.hparams.pretrained_weights_path is not None or mt == "bioclip":
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.frozen_modules.append(self.encoder)

        # Build projection head layers progressively
        current_dim = backbone_dim
        projection_layers: list[nn.Module] = [nn.BatchNorm1d(current_dim), self.activation_function]
        for dim in self.hparams.projection_layer_dims:
            projection_layers.extend(
                [nn.Linear(current_dim, dim),
                 nn.BatchNorm1d(dim),
                 self.activation_function
                 ]
            )
            current_dim = dim

        self.encoder = nn.Sequential(self.encoder, *projection_layers[:-2])
        self.rms = nn.RMSNorm(current_dim, elementwise_affine=False)
        if self.hparams.feature_scaling and not isinstance(self.manifold, Euclidean):
            c: torch.Tensor = self.manifold.c  # type: ignore
            self.feature_scaling_layer = FeatureScaling(current_dim, c, dtype=self.manifold.dtype)

        # Add hyperbolic encoding layer(s)
        # Note: Euclidean manifold will default to nn.Linear
        self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
            layer_name=self.hparams.hyperbolic_encoding,
            manifold=self.manifold,
            input_dim=current_dim,
            output_dim=self.hparams.latent_dim,
            params_dtype=self.hparams.params_dtype,
        )
        self.train()  # frozen modules go to eval mode

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x_aug_1, x_aug_2 = (batch[view] for view in PRETRAINING_VIEWS)

        h_aug_1 = self.rms(self.encoder(x_aug_1))
        h_aug_2 = self.rms(self.encoder(x_aug_2))
        if not torch.isfinite(h_aug_1).all() or not torch.isfinite(h_aug_2).all():
            raise ValueError("Non-finite values encountered in backbone features.")

        if self.hparams.contrastive_space == "euclidean":
            # Cosine-similarity contrastive loss on the projection-head features: the hyperbolic
            # head is skipped during pretraining, since it has enough capacity to minimize the
            # contrastive loss without the backbone improving (gradient absorption), and is
            # trained from scratch during the clustering stage.
            loss = self._compute_euclidean_contrastive_loss(
                h_aug_1, h_aug_2, temperature=self.hparams.contrastive_temperature
            )
            z_aug_1 = h_aug_1
        else:
            # Geodesic contrastive loss on the manifold embeddings, i.e. the contrastive loss of the
            # clustering stage; this also trains the feature scaling and the hyperbolic layer.
            # The manifold operations are kept out of a possible autocast of the pretraining stage.
            with torch.autocast(device_type=h_aug_1.device.type, enabled=False):
                params_dtype = get_dtype_from_precision(self.hparams.params_dtype)
                z_aug_1 = self.embed_features(h_aug_1.to(dtype=params_dtype))
                z_aug_2 = self.embed_features(h_aug_2.to(dtype=params_dtype))
                loss, _ = self.compute_contrastive_loss(
                    z_aug_1, z_aug_2, temperature=self.hparams.contrastive_temperature
                )

        # Logging
        self.metrics.setdefault(f"pretrain/{self.hparams.contrastive_space}_contrastive_loss", []).append(loss.detach().item())
        self.metrics.setdefault("pretrain/total_loss", []).append(loss.detach().item())
        self.metrics.setdefault("pretrain/c", []).append(
            self.manifold.c.detach().item()  # type: ignore
            if self.manifold.c.numel() == 1  # type: ignore
            else self.manifold.c.detach().cpu()  # type: ignore
        )  # type: ignore
        self.metrics.setdefault("latent_space/norm_mean", []).append(z_aug_1.norm(p=2, dim=-1).mean().detach().item())
        self.log(
            "loss",
            loss,
            logger=False,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=False,
        )

        # Safety net: replace NaN/Inf with 0 while preserving the computation graph
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected at epoch {self.current_epoch}, batch {batch_idx}. Replacing with 0.", flush=True)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss
