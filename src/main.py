import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms  # type: ignore
from torchvision.models.resnet import ResNet18_Weights  # type: ignore


def validate_folder_path(path: str) -> None:
    """
    Validates the given folder path.

    Parameters:
        path (str): The path to the folder to be validated.

    Raises:
        FileNotFoundError: If the path does not exist.
        NotADirectoryError: If the path is not a directory.
        PermissionError: If the user does not have read permissions for the directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path '{path}' does not exist.")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"The path '{path}' is not a directory.")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"The path '{path}' is not readable.")


class ImageData:
    """
    This class represents a set of images.
    """

    def __init__(self, dir: str) -> None:
        """
        Initiate a new object by providing a path to a directory containing images.

        Args:
            dir (str): Path to a directory containing images in PNG or JPG format.
        """
        validate_folder_path(dir)
        self.d = dir

    def load_images(self) -> list[Image.Image]:
        """
        Loads and returns the list of images from the object's directory.

        Returns:
            list[Image.Image]: list of images from the object's directory.
        """
        imgs = []
        for f in os.listdir(self.d):
            if f.endswith(".jpg") or f.endswith(".png"):
                imgs.append(Image.open(os.path.join(self.d, f)))
        return imgs


class ImgProcess:
    """
    Class for preprocessing images before prediction.
    """

    def __init__(self, size: int) -> None:
        """
        Args:
            size (int): Target size the images will be resized to.
        """
        self.s = size

    def resize_and_gray(self, img_list: list[Image.Image]) -> list[torch.Tensor]:
        """
        Resize input images and convert them to black and white.

        Args:
            img_list (list[Image.Image]): Input images.

        Returns:
            list[torch.Tensor]: Processed images to feed to a PyTorch model.
        """
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
    """
    A simple image classifier.
    """

    def __init__(self) -> None:
        """
        Instantiate the classification model.
        """
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def predict_img(self, processed_images: list[torch.Tensor]) -> list[int]:
        """
        Classify a list of pre-processed images.

        Args:
            processed_images (list[torch.Tensor]): List of images pre-processed with ImgProcess.resize_and_gray()

        Returns:
            list[int]: IDs of predicted classes.
        """
        results = []
        for img_tensor in processed_images:
            pred = self.mdl(img_tensor.unsqueeze(0))
            pred_class = int(torch.argmax(pred, dim=1).item())
            results.append(pred_class)
        return results


if __name__ == "__main__":
    loader = ImageData("images/")
    images = loader.load_images()

    processor = ImgProcess(256)
    processed_images = processor.resize_and_gray(images)

    with open(Path("assets") / "imagenet1000_clsidx_to_labels.json") as labels_file:
        labels = json.load(labels_file)

    pred = Predictor()
    results = pred.predict_img(processed_images)
    result_labels = [labels[str(res)] for res in results]
    for res, res_label in zip(results, result_labels):
        print(res, res_label)
