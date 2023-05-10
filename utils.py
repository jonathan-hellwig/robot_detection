import itertools
import torch.nn.functional as F
import torch

NUM_BOX_PARAMETERS: int = 4


def xywh_to_tlbr(xywh_bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (center_x, center_y, width, height) format to (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.

    Prameters:
    - xywh_bounding_boxes: (batch_size, 4)
    """
    tlbr_bounding_boxes = torch.zeros_like(xywh_bounding_boxes)
    tlbr_bounding_boxes[:, 0:2] = (
        xywh_bounding_boxes[:, 0:2] - xywh_bounding_boxes[:, 2:4] / 2
    )

    tlbr_bounding_boxes[:, 2:4] = (
        xywh_bounding_boxes[:, 0:2] + xywh_bounding_boxes[:, 2:4] / 2
    )
    return tlbr_bounding_boxes


class Encoder:
    def __init__(
        self,
        default_box_scalings: torch.Tensor,
        classes: list[str],
        feature_map_size: tuple[int, int] = (10, 8),
        iou_threshold: float = 0.5,
    ) -> None:
        self.default_scalings = default_box_scalings
        self.feature_map_width, self.feature_map_height = feature_map_size
        self.classes = classes
        self.num_classes = len(classes)
        self.threshold = iou_threshold
        self.default_boxes_tl_br = self._default_boxes("tlbr")
        self.default_boxes_xy_wh = self._default_boxes("xywh")

    def apply(
        self, target_boxes: torch.Tensor, target_classes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if target_boxes.size(0) == 0:
            num_default_boxes = self.default_boxes_xy_wh.size(0)
            return (
                torch.zeros((num_default_boxes, NUM_BOX_PARAMETERS)),
                torch.zeros((num_default_boxes,), dtype=torch.bool),
                torch.zeros(num_default_boxes, dtype=torch.long),
            )
        target_boxes_tl_br = xywh_to_tlbr(target_boxes)
        # Select the default box with the highest IoU and with IoU higher than the threshold value
        ious = intersection_over_union(target_boxes_tl_br, self.default_boxes_tl_br)
        _, best_dbox_idx = ious.max(dim=1)
        masked_ious = (
            torch.logical_or(
                (
                    best_dbox_idx
                    == torch.arange(0, self.default_boxes_tl_br.size(0)).reshape(-1, 1)
                ).T,
                ious > self.threshold,
            )
            * ious
        )
        # Select the target box with the highest IoU for each default box
        best_value, best_idx = masked_ious.max(dim=0)

        is_object = best_value > 0
        selected_target_boxes = target_boxes[best_idx[is_object]]
        selected_default_boxes = self.default_boxes_xy_wh[is_object]

        # Encode target boxes with relative offsets to default box
        encoded_target_boxes = self.encode_bounding_boxes(
            selected_target_boxes, selected_default_boxes, is_object
        )

        encoded_target_classes = torch.zeros(
            self.default_boxes_xy_wh.size(0), dtype=torch.long
        )

        encoded_target_classes[is_object] = target_classes[
            best_idx[is_object]
        ].flatten()
        return encoded_target_boxes, is_object, encoded_target_classes

    def encode_bounding_boxes(
        self, selected_target_boxes, selected_default_boxes, is_object
    ):
        encoded_target_boxes = torch.zeros(
            (self.default_boxes_xy_wh.size(0), NUM_BOX_PARAMETERS)
        )
        encoded_target_boxes[is_object, 0:2] = (
            selected_target_boxes[:, 0:2] - selected_default_boxes[:, 0:2]
        ) / selected_default_boxes[:, 2:4]
        encoded_target_boxes[is_object, 2:4] = torch.log(
            selected_target_boxes[:, 2:4] / selected_default_boxes[:, 2:4]
        )
        return encoded_target_boxes

    def decode_model_output(
        self, predicted_boxes: torch.Tensor, encoded_predicted_classes: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode model output using the encoder. The decoded boxes are in (cx, cy, w, h) format.

        Prameters:
        - predicted_boxes: raw model output with shape (batch_size, feature_map_width * feature_map_height, 4)
        """
        assert predicted_boxes.dim() == 3
        prediction_is_object = encoded_predicted_classes > 0
        decoded_boxes = torch.zeros_like(predicted_boxes)
        decoded_boxes[:, :, 0:2] = (
            self.default_boxes_xy_wh[:, 2:4] * (predicted_boxes[:, :, 0:2])
            + self.default_boxes_xy_wh[:, 0:2]
        )
        decoded_boxes[:, :, 2:4] = self.default_boxes_xy_wh[:, 2:4] * torch.exp(
            predicted_boxes[:, :, 2:4]
        )
        decoded_boxes = decoded_boxes.reshape((-1, NUM_BOX_PARAMETERS))
        return (decoded_boxes, encoded_predicted_classes, prediction_is_object)

    def _default_boxes(self, type: str):
        assert type in ["xywh", "tlbr"]
        default_boxes = torch.zeros(
            (
                self.feature_map_width,
                self.feature_map_height,
                self.default_scalings.size(0),
                NUM_BOX_PARAMETERS,
            )
        )
        for i, j in itertools.product(
            range(self.feature_map_width), range(self.feature_map_height)
        ):
            center = torch.tensor(
                [
                    (i + 0.5) / self.feature_map_width,
                    (j + 0.5) / self.feature_map_height,
                ]
            )
            if type == "xywh":
                default_boxes[i, j, :, 0:2] = center
                default_boxes[i, j, :, 2:4] = self.default_scalings
            else:
                default_boxes[i, j, :, 0:2] = center - self.default_scalings / 2
                default_boxes[i, j, :, 2:4] = center + self.default_scalings / 2

        return default_boxes.reshape((-1, NUM_BOX_PARAMETERS))


def intersection_over_union(
    boxes_1: torch.Tensor, boxes_2: torch.Tensor
) -> torch.Tensor:
    """
    Calculation of pairwise intersection over union metric based on two boxes tensor.
    Reference: https://github.com/kuangliu/pytorch-src.

    Paramters:
        - boxes_1 with shape (N, 4)
        - boxes_2 with shape (M, 4)

    output:
        IoU (N, M)
    """
    N = boxes_1.size(0)
    M = boxes_2.size(0)

    be1 = boxes_1.unsqueeze(1).expand(-1, M, -1)
    be2 = boxes_2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    top_left = torch.max(be1[:, :, :2], be2[:, :, :2])
    bottom_right = torch.min(be1[:, :, 2:], be2[:, :, 2:])

    delta = bottom_right - top_left
    delta[delta < 0] = 0
    intersect = delta[:, :, 0] * delta[:, :, 1]

    delta1 = be1[:, :, 2:] - be1[:, :, :2]
    area1 = delta1[:, :, 0] * delta1[:, :, 1]
    delta2 = be2[:, :, 2:] - be2[:, :, :2]
    area2 = delta2[:, :, 0] * delta2[:, :, 1]

    iou = intersect / (area1 + area2 - intersect)
    return iou


def calculate_predicted_classes(predicted_class_logits: torch.Tensor) -> torch.Tensor:
    class_probabilities = F.softmax(predicted_class_logits, dim=-1)
    return torch.argmax(class_probabilities, dim=-1)
