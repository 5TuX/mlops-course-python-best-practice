import os

import torch
from PIL import Image
from torchvision import models, transforms  # type: ignore
from torchvision.models.resnet import ResNet18_Weights  # type: ignore


class imageData:
    def __init__(self, DIR: str) -> None:
        self.D = DIR

    def LoadImages(self) -> list[Image.Image]:
        imgs = []
        for F in os.listdir(self.D):
            if F.endswith(".jpg") or F.endswith(".png"):
                imgs.append(Image.open(os.path.join(self.D, F)))
        return imgs


class imgProcess:
    def __init__(self, size: int) -> None:
        self.s = size

    def resize_and_GRAY(self, img_list: list[Image.Image]) -> list[torch.Tensor]:
        p_images = []
        for img in img_list:
            t = transforms.Compose(
                [
                    transforms.Resize((self.s, self.s)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            p_images.append(t(img))
        return p_images


class Predictor:
    def __init__(self) -> None:
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def Predict_Img(self, processed_images: list[torch.Tensor]) -> list[int | float]:
        results = []
        for img_tensor in processed_images:
            pred = self.mdl(img_tensor.unsqueeze(0))
            results.append(torch.argmax(pred, dim=1).item())
        return results


if __name__ == "__main__":
    loader = imageData("images/")
    images = loader.LoadImages()

    processor = imgProcess(256)
    processed_images = processor.resize_and_GRAY(images)

    pred = Predictor()
    results = pred.Predict_Img(processed_images)
    print(results)
