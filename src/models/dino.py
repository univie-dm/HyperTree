import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hyperbolic_math.src.manifolds import Euclidean
from src.models.resnet import HyperbolicResNet


class DINOHead(nn.Module):
    """DINO projection head: MLP + L2-norm + weight-normalized last layer."""

    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        # Manual weight normalization (avoids nn.utils.parametrizations which breaks pickling)
        w = self.last_layer.weight
        x = F.linear(x, F.normalize(w, dim=1, p=2))
        return x


class DINOHyperbolicResNet(HyperbolicResNet):
    """
    DINO self-distillation pretraining that produces a HyperbolicResNet-compatible model.

    The student is the standard HyperbolicResNet encoder.
    The teacher is an EMA copy. Centering prevents collapse.
    After pretraining, this model has the same encoder/rms/feature_scaling_layer/
    hyperbolic_encoding_layer structure that HyphC expects.
    """

    def __init__(
        self,
        # DINO-specific parameters
        dino_out_dim: int = 65536,
        dino_hidden_dim: int = 2048,
        dino_bottleneck_dim: int = 256,
        teacher_momentum_start: float = 0.996,
        teacher_momentum_end: float = 1.0,
        teacher_temperature: float = 0.04,
        student_temperature: float = 0.1,
        center_momentum: float = 0.9,
        n_local_crops: int = 6,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        max_epochs: int = 100,
        # All HyperbolicResNet parameters
        **kwargs,
    ):
        # Force no supervised weights for DINO
        kwargs["load_supervised_weights"] = False
        kwargs["supervised"] = False
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Get backbone output dimension (before hyperbolic encoding)
        # The encoder ends at the projection head; get its output dim
        # from the last linear layer in the encoder Sequential
        backbone_out_dim = self._get_encoder_out_dim()

        # Student DINO head (operates on backbone features before hyperbolic encoding)
        self.dino_head_student = DINOHead(
            in_dim=backbone_out_dim,
            out_dim=dino_out_dim,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
        )

        # Teacher: full copy of encoder + rms + its own DINO head
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_rms = copy.deepcopy(self.rms)
        self.dino_head_teacher = DINOHead(
            in_dim=backbone_out_dim,
            out_dim=dino_out_dim,
            hidden_dim=dino_hidden_dim,
            bottleneck_dim=dino_bottleneck_dim,
        )
        # Initialize teacher head from student head
        self.dino_head_teacher.load_state_dict(self.dino_head_student.state_dict())

        # Freeze teacher
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_rms.parameters():
            p.requires_grad = False
        for p in self.dino_head_teacher.parameters():
            p.requires_grad = False

        # Center buffer for teacher outputs
        self.register_buffer("center", torch.zeros(1, dino_out_dim))

        # Teacher temperature schedule
        self.register_buffer(
            "teacher_temp_schedule",
            torch.linspace(warmup_teacher_temp, teacher_temperature, warmup_teacher_temp_epochs),
        )

    def on_train_epoch_start(self):
        """Freeze the last layer of the student DINO head for the first epoch to prevent collapse."""
        if self.current_epoch == 0:
            self.dino_head_student.last_layer.weight.requires_grad = False
        elif self.current_epoch == 1:
            self.dino_head_student.last_layer.weight.requires_grad = True

    def _get_encoder_out_dim(self) -> int:
        """Get the output dimension of the encoder (before RMS/hyperbolic)."""
        # Walk backwards through the Sequential to find the last Linear or BatchNorm1d
        for module in reversed(list(self.encoder.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
            if isinstance(module, nn.BatchNorm1d):
                return module.num_features
        raise RuntimeError("Could not determine encoder output dimension")

    @torch.no_grad()
    def _update_teacher(self):
        """EMA update of teacher from student."""
        epoch = self.current_epoch
        max_epochs = self.hparams.max_epochs
        # Cosine momentum schedule
        m_start = self.hparams.teacher_momentum_start
        m_end = self.hparams.teacher_momentum_end
        m = m_end - (m_end - m_start) * (1 + torch.cos(torch.tensor(torch.pi * epoch / max_epochs))) / 2
        m = m.item()

        for ps, pt in zip(self.encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.mul_(m).add_((1 - m) * ps.data)
        for ps, pt in zip(self.rms.parameters(), self.teacher_rms.parameters()):
            pt.data.mul_(m).add_((1 - m) * ps.data)
        for ps, pt in zip(self.dino_head_student.parameters(), self.dino_head_teacher.parameters()):
            pt.data.mul_(m).add_((1 - m) * ps.data)

    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor):
        """Update center with EMA of teacher outputs."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.hparams.center_momentum + batch_center * (1 - self.hparams.center_momentum)

    def _get_teacher_temp(self) -> float:
        epoch = self.current_epoch
        if epoch < len(self.teacher_temp_schedule):
            return self.teacher_temp_schedule[epoch].item()
        return self.hparams.teacher_temperature

    def _student_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Student forward: encoder -> rms -> dino_head. Returns (dino_output, backbone_features)."""
        h = self.encoder(x)
        h = self.rms(h)
        return self.dino_head_student(h), h

    @torch.no_grad()
    def _teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher forward: teacher_encoder -> teacher_rms -> dino_head_teacher."""
        h = self.teacher_encoder(x)
        h = self.teacher_rms(h)
        return self.dino_head_teacher(h)

    def dino_loss(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss between student and teacher softmax outputs.
        Teacher outputs are centered and sharpened.
        """
        teacher_temp = self._get_teacher_temp()
        student_temp = self.hparams.student_temperature

        # Teacher: center + sharpen
        teacher_probs = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1).detach()
        # Student: just temperature
        student_log_probs = F.log_softmax(student_out / student_temp, dim=-1)

        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # batch: (global1, global2, base_img, label, local1, ..., localN)
        global1 = batch[0]
        global2 = batch[1]
        # base_img = batch[2], label = batch[3]
        local_crops = list(batch[4:])

        all_crops = [global1, global2] + local_crops
        n_global = 2

        # Student forward on ALL crops — also returns backbone features
        student_results = [self._student_forward(crop) for crop in all_crops]
        student_outputs = [r[0] for r in student_results]
        # Save backbone features from global crops for hyperbolic loss (avoids re-encoding)
        h_aug_1 = student_results[0][1]
        h_aug_2 = student_results[1][1]

        # Teacher forward on GLOBAL crops only
        with torch.no_grad():
            teacher_outputs = [self._teacher_forward(crop) for crop in all_crops[:n_global]]
            teacher_cat = torch.cat(teacher_outputs, dim=0)

        # Compute DINO loss: each teacher global crop vs each student crop (excluding same crop)
        total_loss = torch.tensor(0.0, device=global1.device)
        n_loss_terms = 0
        for t_idx in range(n_global):
            for s_idx in range(len(all_crops)):
                if t_idx == s_idx:
                    continue  # Skip same crop
                total_loss += self.dino_loss(student_outputs[s_idx], teacher_outputs[t_idx])
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update teacher (EMA) and center
        self._update_teacher()
        self._update_center(teacher_cat)

        # Log DINO-only loss before adding Euclidean contrastive
        self.metrics.setdefault("pretrain/dino_loss", []).append(total_loss.detach().item())

        # Euclidean contrastive only — skip hyperbolic head during pretraining.
        # Same rationale as SimCLR: the hyperbolic head absorbs gradient without
        # improving the backbone. DINO self-distillation + Euclidean contrastive
        # train the backbone; the hyperbolic head is trained during clustering.
        euclidean_contrastive_loss = self._compute_euclidean_contrastive_loss(
            h_aug_1, h_aug_2, temperature=self.hparams.contrastive_temperature
        )
        c_weight = self.hparams.contrastive_weight if self.hparams.contrastive_loss_enabled else 0.1
        total_loss = total_loss + c_weight * euclidean_contrastive_loss

        self.metrics.setdefault("pretrain/euclidean_contrastive_loss", []).append(euclidean_contrastive_loss.detach().item())
        z_aug_1 = h_aug_1  # backbone features for latent space stats below

        # Logging
        self.metrics.setdefault("pretrain/total_loss", []).append(total_loss.detach().item())
        self.metrics.setdefault("pretrain/teacher_temp", []).append(self._get_teacher_temp())
        self.metrics.setdefault("pretrain/c", []).append(
            self.manifold.c.detach().item()
            if self.manifold.c.numel() == 1
            else self.manifold.c.detach().cpu()
        )
        self.log("loss", total_loss, logger=False, prog_bar=True, on_epoch=True, on_step=False, sync_dist=False)

        # Latent space stats from global crop (reuse z_aug_1 computed above)
        with torch.no_grad():
            z = z_aug_1.detach()
            self.metrics.setdefault("latent_space/mean", []).append(z.mean().item())
            self.metrics.setdefault("latent_space/std", []).append(z.std().item())
            self.metrics.setdefault("latent_space/max", []).append(z.max().item())
            self.metrics.setdefault("latent_space/min", []).append(z.min().item())
            self.metrics.setdefault("latent_space/norm_mean", []).append(z.norm(p=2, dim=-1).mean().item())

        # Safety net
        if not torch.isfinite(total_loss):
            print(f"Warning: Non-finite loss at epoch {self.current_epoch}, batch {batch_idx}. Replacing with 0.", flush=True)
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return total_loss
