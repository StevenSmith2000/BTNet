import torch
import torch.nn.functional as F

from ._base import Distiller

class Distill(Distiller):

    def __init__(self, student, teacher, cfg, downsample):
        super(Distill, self).__init__(student, teacher)
        self.distill_feat_index = cfg.distill_feat_index
        self.downsample = downsample
        self.feat_loss_weight = cfg.feat_loss_weight
        self.embed_loss_weight = cfg.embed_loss_weight

    def get_learnable_parameters(self):
        return super().get_learnable_parameters()

    def forward_train(self, image):
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        image = self.downsample(image)

        logits_student, feature_student = self.student(image)

        loss_feat = self.feat_loss_weight * F.mse_loss(
            feature_student[self.distill_feat_index], feature_teacher[self.distill_feat_index]
        )
        loss_embed = self.embed_loss_weight * F.mse_loss(
                logits_student, logits_teacher
            )
        losses_dict = {
            "loss_feat": loss_feat,
            "loss_kd": loss_embed,
        }

        return logits_student, losses_dict
