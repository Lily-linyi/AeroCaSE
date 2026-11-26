"""AeroCaSE segmentation and auxiliary supervision."""
import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from rein.models.heads.rein_mask2former import ReinMask2FormerHead


@MODELS.register_module()
class AeroCaSEHead(ReinMask2FormerHead):
    """Segmentation head with prior, alignment, proxy and hard-pixel losses."""

    def __init__(
        self, *, small_obj_ids, loss_so_weight, loss_hier_weight,
        loss_proxy_weight, loss_soem_weight, soem_ratio, soem_threshold,
        small_class_weight, ignore_index=255, **kwargs,
    ):
        super().__init__(**kwargs)
        self.small_obj_ids = tuple(int(i) for i in small_obj_ids)
        self.ignore_index = int(ignore_index)
        self.loss_so_weight = float(loss_so_weight)
        self.loss_hier_weight = float(loss_hier_weight)
        self.loss_proxy_weight = float(loss_proxy_weight)
        self.loss_soem_weight = float(loss_soem_weight)
        self.soem_ratio = float(soem_ratio)
        self.soem_threshold = float(soem_threshold)
        self.small_class_weight = float(small_class_weight)

    def _split_backbone_output(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 3:
            feats, query_embed, aux_small_logits = x
            return (feats, query_embed), aux_small_logits
        return x, None

    def forward(self, x, batch_data_samples):
        x_for_rein, _ = self._split_backbone_output(x)
        return super().forward(x_for_rein, batch_data_samples)

    def _stack_gt(self, batch_data_samples, size, device):
        labels = []
        for sample in batch_data_samples:
            label = sample.gt_sem_seg.data.squeeze(0).long()
            labels.append(label)
        labels = torch.stack(labels, dim=0).to(device)
        if labels.shape[-2:] != size:
            labels = F.interpolate(
                labels.unsqueeze(1).float(), size=size, mode="nearest"
            ).squeeze(1).long()
        return labels

    def _small_object_target(self, labels):
        target = torch.zeros_like(labels, dtype=torch.long)
        target[labels == self.ignore_index] = self.ignore_index
        for class_id in self.small_obj_ids:
            target[labels == class_id] = 1
        return target

    def _semantic_prob_from_masks(self, cls_score, mask_pred):
        class_prob = F.softmax(cls_score, dim=-1)[..., : self.num_classes]
        mask_prob = mask_pred.sigmoid()
        sem_prob = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
        sem_prob = sem_prob / sem_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return sem_prob.clamp_min(1e-6)

    @staticmethod
    def _masked_mean(loss_map, valid):
        if valid.any():
            return loss_map[valid].mean()
        return loss_map.sum() * 0.0

    def _soem(self, loss_map, labels, small_target):
        valid = labels != self.ignore_index
        n_min = max(1, int(valid.sum().item() * self.soem_ratio))

        def pick(mask):
            values = loss_map[mask]
            if values.numel() == 0:
                return values
            hard = values[values > self.soem_threshold]
            if hard.numel() < n_min:
                k = min(n_min, values.numel())
                hard = values.topk(k).values
            return hard

        hard_small = pick(valid & (small_target == 1))
        hard_large = pick(valid & (small_target == 0))
        count = hard_small.numel() + hard_large.numel()
        if count == 0:
            return loss_map.sum() * 0.0
        return (hard_small.sum() + hard_large.sum()) / count

    def loss(self, x, batch_data_samples, train_cfg):
        x_for_rein, aux_small_logits = self._split_backbone_output(x)
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(batch_data_samples)
        all_cls_scores, all_mask_preds = self.forward(x_for_rein, batch_data_samples)
        losses = self.loss_by_feat(
            all_cls_scores, all_mask_preds, batch_gt_instances, batch_img_metas
        )
        if aux_small_logits is None:
            return losses

        labels = self._stack_gt(batch_data_samples, aux_small_logits.shape[-2:], aux_small_logits.device)
        small_target = self._small_object_target(labels)
        class_weight = aux_small_logits.new_tensor([1.0, self.small_class_weight])
        so_ce_map = F.cross_entropy(
            aux_small_logits,
            small_target,
            weight=class_weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = labels != self.ignore_index
        losses["loss_sos_top"] = self.loss_so_weight * self._masked_mean(so_ce_map, valid)

        sem_prob = self._semantic_prob_from_masks(all_cls_scores[-1], all_mask_preds[-1])
        if sem_prob.shape[-2:] != aux_small_logits.shape[-2:]:
            sem_prob = F.interpolate(
                sem_prob, size=aux_small_logits.shape[-2:], mode="bilinear", align_corners=False
            )
            sem_prob = sem_prob / sem_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)

        small_ids = [i for i in self.small_obj_ids if 0 <= i < self.num_classes]
        large_ids = [i for i in range(self.num_classes) if i not in small_ids]
        bottom_large = sem_prob[:, large_ids].sum(dim=1, keepdim=True)
        bottom_small = sem_prob[:, small_ids].sum(dim=1, keepdim=True)
        bottom_top = torch.cat([bottom_large, bottom_small], dim=1)
        top_prob = F.softmax(aux_small_logits, dim=1)
        hier_map = (bottom_top - top_prob).pow(2).sum(dim=1)
        losses["loss_sos_hier"] = self.loss_hier_weight * self._masked_mean(hier_map, valid)

        proxy_labels = self._stack_gt(batch_data_samples, sem_prob.shape[-2:], sem_prob.device)
        proxy_map = F.nll_loss(
            torch.log(sem_prob),
            proxy_labels,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        losses["loss_sos_proxy"] = self.loss_proxy_weight * self._masked_mean(
            proxy_map, proxy_labels != self.ignore_index
        )

        point_loss = so_ce_map + self.loss_hier_weight * hier_map
        losses["loss_sos_soem"] = self.loss_soem_weight * self._soem(
            point_loss, labels, small_target
        )
        return losses
