"""
from numpy import save
from src.models.abstract import AbstractModel
from src.models.utils import get_dtype_from_precision, get_hyperbolic_decoding_layer_by_name, get_hyperbolic_encoding_layer_by_name
from src.evaluation import evaluation
from src.hyperbolic_math.src.utils.helpers import compute_pairwise_distances
from src.hyperbolic_math.src.manifolds import Euclidean, Hyperboloid, PoincareBall
from src.hyperbolic_math.src.optim.riemannian_adam import RiemannianAdam
from src.hyperbolic_math.src.optim.riemannian_sgd import RiemannianSGD
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm


from src.poincare_resnet.models.resnets.hyperbolic import HyperbolicResNet
from src.models.resnet import DecoderBlock, resize_conv1x1 # Re-use from the existing resnet

class PoincareResNet(AbstractModel):
    def __init__(self,
                 input_channels: int,
                 params_dtype: torch.dtype,
                 latent_dim: int,
                 manifold_name: str,
                 manifold_precision: str,
                 curvature_magnitude: float,
                 trainable_curvature: bool,
                 optimizer_name: str,
                 learning_rate: float,
                 log_interval: int,
                 distances_type: str,
                 hyperbolic_encoding: str,
                 hyperbolic_decoding: str,
                 weight_decay: float = 0.0,
                 evaluation_metrics: dict[str, bool] = {},
                 activation_function: str = 'relu',
                 clamping_factor: float = 1.0,
                 smoothing_factor: float = 50.0,
                 learnable_scale: bool = False,
                 normalize: bool = False
                 ):
        super().__init__(
            model_type='poincare_resnet_ae',
            hyperbolic_encoding=hyperbolic_encoding,
            params_dtype=params_dtype,
            latent_dim=latent_dim,
            manifold_name=manifold_name,
            manifold_precision=manifold_precision,
            curvature_magnitude=curvature_magnitude,
            trainable_curvature=trainable_curvature,
            hyperbolic_decoding=hyperbolic_decoding,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            log_interval=log_interval,
            distances_type=distances_type,
            evaluation_metrics=evaluation_metrics,
            activation_function=activation_function,
        )
        self.save_hyperparameters() 
        self.model_type = 'poincare_resnet_ae'
        self.params_dtype = params_dtype
        self.manifold_name = manifold_name
        self.manifold_precision = manifold_precision
        self.curvature_magnitude = curvature_magnitude
        self.trainable_curvature = trainable_curvature
        self.distances_type = distances_type
        self.global_epoch = 0
        self.metrics = {}
        self.stage = 'pretrain'
        
        self.latent_dim = latent_dim
        self.evaluation_metrics = evaluation_metrics
        self.log_interval = log_interval
        self.input_channels = input_channels
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clamping_factor = clamping_factor
        self.smoothing_factor = smoothing_factor
        self.learnable_scale = learnable_scale
        self.normalize = normalize

        channel_dims = [16, 32, 64]
        depths = [3,3,3]
        self.inplanes = 64
        
        if isinstance(self.manifold, Euclidean) or hyperbolic_encoding is None:
            print("Using Euclidean manifold or hyperbolic_encoding is 'None', "
                  "hyperbolic encoding layers will default to nn.Linear.", flush=True)
            self.hyperbolic_encoding_layer = nn.Linear(channel_dims[-1], self.latent_dim)
        else:
            self.hyperbolic_encoding_layer = get_hyperbolic_encoding_layer_by_name(
                layer_name=hyperbolic_encoding,
                manifold=self.manifold,
                input_dim=channel_dims[-1],
                output_dim=self.latent_dim,
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                learnable_scale=self.learnable_scale,
                normalize=self.normalize
            )

        if isinstance(self.manifold, Euclidean) or hyperbolic_decoding is None:
            print("Using Euclidean manifold or hyperbolic_encoding is 'None', "
                  "hyperbolic encoding layers will default to nn.Linear.", flush=True)
            self.hyperbolic_decoding_layer = nn.Sequential(
                spectral_norm(nn.Linear(self.latent_dim, channel_dims[-1])),
                nn.BatchNorm1d(channel_dims[-1]),
                self.activation_function
            )
        else:
            self.hyperbolic_decoding_layer = get_hyperbolic_decoding_layer_by_name(
                layer_name=hyperbolic_decoding,
                manifold=self.manifold,
                input_dim=self.latent_dim,
                output_dim=(self.inplanes * 8 * 8),
                params_dtype=self.params_dtype,
                clamping_factor=self.clamping_factor,
                smoothing_factor=self.smoothing_factor,
                learnable_scale=self.learnable_scale,
                normalize=self.normalize
            )

        self.hyperbolic_encoding = hyperbolic_encoding
        self.hyperbolic_decoding = hyperbolic_decoding

        base_model = HyperbolicResNet(
            classes=latent_dim,
            channel_dims=channel_dims,
            depths=depths,
        )

        self.encoder = nn.Sequential(
            base_model.conv,
            base_model.bn,
            base_model.act,
            base_model.group1,
            base_model.group2,
            base_model.group3,
            base_model.avg_pool,
            nn.Flatten(),
        )

        
        # The hyperbolic layer is now an identity map during pre-training.
        # It's only used for its manifold properties during the clustering stage.
        self.hyperbolic_layer = nn.Identity()
        
        # --- BUILD THE DECODER ---
        self.inplanes = channel_dims[-1]
        

        self.decoder = nn.Sequential(
            self._make_decoder_layer(128, 2, scale=2), # Note: Adjusted these channels for a better capacity decoder
            self._make_decoder_layer(64, 2, scale=2),
            nn.ConvTranspose2d(self.inplanes, self.input_channels, kernel_size=3, stride=1, padding=1),
            self.activation_function
        )

    def _make_decoder_layer(self, planes: int, blocks: int, scale: int = 1) -> nn.Sequential:
        upsample = None
        if scale != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes, scale),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(DecoderBlock(self.inplanes, planes, scale, upsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DecoderBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.hyperbolic_encoding_layer(z)
        if isinstance(self.manifold, Hyperboloid):
            raise NotImplementedError("Hyperboloid manifold is not supported by the PoincareResNet.")
        return self.hyperbolic_layer(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Reshape for decoder
        z = self.hyperbolic_decoding_layer(z)
        z = z.view(z.size(0), -1, 8, 8) # Reshape to 4D tensor (batch, channels, H, W)
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_z = self.encode(x)
        reconstructed_x = self.decode(latent_z)
        return reconstructed_x, latent_z


    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        self.orginal_input_shape = x.shape[1:]
        x_hat, z = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.metrics.setdefault("pretrain/recon_loss", []).append(loss.item())
        self.metrics['latent_space/mean'] = z.mean().item()
        self.metrics['latent_space/std'] = z.std().item()
        self.metrics['latent_space/max'] = z.max().item()
        self.metrics['latent_space/min'] = z.min().item()
        if batch_idx == 0:
            from torchvision.utils import save_image
            save_image(x[0], 'original.png')
            save_image(x_hat[0], 'reconstructed.png')
        return loss
"""