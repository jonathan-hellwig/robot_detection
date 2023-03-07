import itertools
import torch.nn.functional as F
import torch

NUM_BOX_PARAMETERS: int = 4


def xywh_to_tlbr(xywh_bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from (cx, cy, w, h) format to (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.

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
        default_scalings: torch.Tensor,
        num_classes: int,
        threshold: float = 0.5,
    ) -> None:
        # width x height
        self.default_scalings = default_scalings
        self.feature_map_width, self.feature_map_height = (10, 8)
        self.num_classes = num_classes
        self.threshold = threshold
        self.default_boxes_tl_br = self._default_boxes("tlbr")
        self.default_boxes_xy_wh = self._default_boxes("xywh")

    def apply(self, target_boxes: torch.Tensor, target_classes: torch.Tensor):
        # Transform bounding boxes from (cx, cy, w, h) to (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
        # TODO: Reduce an error source by refactoring this section
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

    def decode_bounding_boxes(self, predicted_boxes: torch.Tensor) -> torch.Tensor:
        """
        Decode model output using the encoder. The decoded boxes are in (cx, cy, w, h) format.

        Prameters:
        - encoder
        - predicted_boxes: raw model output with shape (batch_size, feature_map_width * feature_map_height, 4)
        """
        assert predicted_boxes.dim() == 3
        decoded_boxes = torch.zeros_like(predicted_boxes)
        decoded_boxes[:, :, 0:2] = (
            self.default_boxes_xy_wh[:, 2:4] * (predicted_boxes[:, :, 0:2])
            + self.default_boxes_xy_wh[:, 0:2]
        )
        decoded_boxes[:, :, 2:4] = self.default_boxes_xy_wh[:, 2:4] * torch.exp(
            predicted_boxes[:, :, 2:4]
        )

        return decoded_boxes

    def _default_boxes(self, type):
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


def count(input: torch.Tensor, num_classes: int) -> torch.Tensor:
    augmented_counts = torch.zeros((num_classes,))
    values, counts = torch.unique(input, return_counts=True)
    assert values.size(0) <= num_classes
    augmented_counts[values - 1] += counts
    return augmented_counts


def mean_average_precision(
    predicted_boxes: torch.Tensor,
    predicted_classes: torch.Tensor,
    target: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    threshold: float,
    num_classes: int,
):
    decoded_target_boxes, target_is_object, target_classes = target

    prediction_is_object = predicted_classes > 0
    object_boxes_tlbr = xywh_to_tlbr(predicted_boxes[prediction_is_object])
    target_boxes_tlbr = xywh_to_tlbr(decoded_target_boxes[target_is_object])

    overlap_score = intersection_over_union(object_boxes_tlbr, target_boxes_tlbr)
    max_overlap_score, max_target_box_index = torch.max(overlap_score, dim=1)
    prediction_matches_object = max_overlap_score > threshold

    # Case 1. object detected, overlap below threshold
    predicted_object_classes = predicted_classes[prediction_is_object]
    false_object_predicition_no_overlap_counts = count(
        predicted_object_classes[~prediction_matches_object], num_classes
    )

    # Case 2. object detected, overlap above theshold, correct_class
    matching_target_box_class = target_classes[target_is_object][
        max_target_box_index[prediction_matches_object]
    ]
    true_object_prediction_correct_class = (
        predicted_object_classes[prediction_matches_object] == matching_target_box_class
    )
    true_object_prediction_counts = count(
        predicted_object_classes[prediction_matches_object][
            true_object_prediction_correct_class
        ],
        num_classes,
    )
    false_object_predicition_overlap_counts = count(
        predicted_object_classes[prediction_matches_object][
            ~true_object_prediction_correct_class
        ],
        num_classes,
    )

    total_object_predicition_counts = (
        false_object_predicition_overlap_counts
        + false_object_predicition_no_overlap_counts
        + true_object_prediction_counts
    )
    object_precisions = true_object_prediction_counts / total_object_predicition_counts
    mean_average_precision = object_precisions.mean()
    return mean_average_precision


# Test case
def equal_distribution():
    num_classes = 3
    # Predicted classes flat list of class label
    predicted_classes = torch.tensor([0, 1, 2, 3])
    # Predicted boxes in format (cx, cy, w, h)
    predicted_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.25, 0.25],
            [0.75, 0.25, 0.25, 0.25],
            [0.25, 0.75, 0.25, 0.25],
            [0.75, 0.75, 0.25, 0.25],
        ]
    )

    target_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.25, 0.25],
            [0.75, 0.25, 0.25, 0.25],
            [0.25, 0.75, 0.25, 0.25],
            [0.75, 0.75, 0.25, 0.25],
        ]
    )
    target_is_object = torch.tensor([False, True, True, True])
    target_classes = torch.tensor([0, 1, 2, 3])
    target = (target_boxes, target_is_object, target_classes)
    threshold = 0.0

    result = mean_average_precision(
        predicted_boxes, predicted_classes, target, threshold, num_classes
    )
    assert torch.allclose(result, torch.tensor([1.0, 1.0, 1.0]))


# Test case
def box_too_small():
    num_classes = 3
    # Predicted classes flat list of class label
    predicted_classes = torch.tensor([0, 1, 2, 3])
    # Predicted boxes in format (cx, cy, w, h)
    predicted_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.25, 0.25],
            [0.75, 0.25, 0.05, 0.05],
            [0.25, 0.75, 0.25, 0.25],
            [0.75, 0.75, 0.25, 0.25],
        ]
    )

    target_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.25, 0.25],
            [0.75, 0.25, 0.25, 0.25],
            [0.25, 0.75, 0.25, 0.25],
            [0.75, 0.75, 0.25, 0.25],
        ]
    )
    target_is_object = torch.tensor([False, True, True, True])
    target_classes = torch.tensor([0, 1, 2, 3])
    target = (target_boxes, target_is_object, target_classes)
    threshold = 0.5

    result = mean_average_precision(
        predicted_boxes, predicted_classes, target, threshold, num_classes
    )
    print(result)
    assert torch.allclose(result, torch.tensor([0.0, 1.0, 1.0]))


def not_equal():
    """
    Predicted boxes:
    | 0 | 1/2 |
    | 2 | 3 |

    Target boxes:
    | 0 | 1 |
    | 2 | 3 |
    """
    num_classes = 3
    # Predicted classes flat list of class label
    predicted_classes = torch.tensor([0, 1, 2, 2, 3])
    # Predicted boxes in format (cx, cy, w, h)
    predicted_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.25, 0.25],
            [0.75, 0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25, 0.25],
            [0.25, 0.75, 0.25, 0.25],
            [0.75, 0.75, 0.25, 0.25],
        ]
    )

    target_boxes = torch.tensor(
        [
            [0.0, 0.0, 0.25, 0.25],
            [0.75, 0.25, 0.25, 0.25],
            [0.25, 0.75, 0.25, 0.25],
            [0.75, 0.75, 0.25, 0.25],
        ]
    )
    target_is_object = torch.tensor([False, True, True, True])
    target_classes = torch.tensor([0, 1, 2, 3])
    target = (target_boxes, target_is_object, target_classes)
    threshold = 0.0

    result = mean_average_precision(
        predicted_boxes, predicted_classes, target, threshold, num_classes
    )
    print(result)
    assert torch.allclose(result, torch.tensor([1.0, 0.5, 1.0]))
