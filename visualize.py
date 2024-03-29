import torchvision.transforms as T
from PIL import ImageDraw, Image
import torch


def tensor_to_pil(image: torch.Tensor, normalize: torch.Tensor):
    image = image * normalize[1] + normalize[0]
    return T.ToPILImage()(image)


def draw_bounding_box(image: Image.Image, bounding_boxes: torch.Tensor) -> Image.Image:
    image_draw = ImageDraw.Draw(image)
    for i in range(bounding_boxes.size(0)):
        cx = int(bounding_boxes[i][0] * image.size[0])
        cy = int(bounding_boxes[i][1] * image.size[1])
        w = int(bounding_boxes[i][2] * image.size[0])
        h = int(bounding_boxes[i][3] * image.size[1])
        xy = [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2]
        image_draw.rectangle(xy, fill=128)
    return image


def image_grid(imgs, rows: int, cols: int) -> Image.Image:
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def draw_model_output(
    image: Image.Image,
    decoded_boxes: torch.Tensor,
    predicted_classes: torch.Tensor,
    normalize: torch.Tensor,
) -> Image.Image:
    detection_is_object = predicted_classes > 0
    object_boxes = decoded_boxes[detection_is_object]

    image = draw_bounding_box(tensor_to_pil(image, normalize), object_boxes)
    return image
