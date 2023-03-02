import torch.nn.functional as F
import torchvision.transforms as T
from PIL import ImageDraw, Image
import torch


def tensor_to_pil(image, normalize):
    image = image * normalize[1] + normalize[0]
    return T.ToPILImage()(image)


def draw_bounding_box(image, bounding_boxes):
    image_draw = ImageDraw.Draw(image)
    for i in range(bounding_boxes.size(0)):
        cx = int(bounding_boxes[i][0] * image.size[0])
        cy = int(bounding_boxes[i][1] * image.size[1])
        w = int(bounding_boxes[i][2] * image.size[0])
        h = int(bounding_boxes[i][3] * image.size[1])
        xy = [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
        image_draw.rectangle(xy)
    return image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def draw_model_output(
    image, predicted_boxes, predicted_class_logits, normalize, encoder
):
    class_probabilities = F.softmax(predicted_class_logits, dim=-1)
    decoded_boxes = torch.zeros(predicted_boxes.shape)
    decoded_boxes[:, :, 0:2] = encoder.default_boxes_xy_wh[:, 2:4] * (
        encoder.default_boxes_xy_wh[:, 0:2] - predicted_boxes[:, :, 0:2]
    )
    decoded_boxes[:, :, 2:4] = encoder.default_boxes_xy_wh[:, 2:4] * torch.exp(
        predicted_boxes[:, :, 2:4]
    )
    decoded_boxes = decoded_boxes.squeeze()
    detection_mask = torch.argmax(class_probabilities, dim=-1) > 0
    object_boxes = decoded_boxes[detection_mask]

    image = draw_bounding_box(tensor_to_pil(image, normalize), object_boxes)
    return image
