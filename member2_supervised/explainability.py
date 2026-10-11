"""Generate Grad-CAM visualizations for ConvNeXt-Tiny."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .dataset import make_transform
    from .model import load_checkpoint
except ImportError:
    from dataset import make_transform
    from model import load_checkpoint


def generate_gradcam(
    model,
    image_tensor,
    device,
    class_index=None,
):
    model.eval()

    activations = {}
    gradients = {}

    # Final ConvNeXt feature block.
    target_layer = model.features[-1][-1]

    forward_handle = target_layer.register_forward_hook(
        lambda module, inputs, output:
            activations.__setitem__("value", output)
    )

    backward_handle = target_layer.register_full_backward_hook(
        lambda module, grad_input, grad_output:
            gradients.__setitem__("value", grad_output[0])
    )

    try:
        model.zero_grad(set_to_none=True)

        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)

        if class_index is None:
            target_class = int(
                probabilities.argmax(dim=1).item()
            )
        else:
            target_class = int(class_index)

        if target_class not in (0, 1):
            raise ValueError("class_index must be 0 or 1.")

        logits[0, target_class].backward()

        feature_maps = activations["value"]
        grads = gradients["value"]

        weights = grads.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = F.relu(
            (weights * feature_maps).sum(dim=1, keepdim=True)
        )

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.detach().float().cpu().numpy()

        cam = (
            (cam - cam.min())
            / max(float(cam.max() - cam.min()), 1e-8)
        )

        return (
            cam,
            target_class,
            probabilities[0].detach().cpu().numpy(),
        )

    finally:
        forward_handle.remove()
        backward_handle.remove()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="gradcam.png")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--class-index",
        type=int,
        choices=[0, 1],
        default=None,
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    image_size = args.image_size or int(
        checkpoint.get("image_size", 224)
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))

    tensor = make_transform(
        image_size=image_size,
        augment=False,
    )(image).unsqueeze(0)

    cam, target, probabilities = generate_gradcam(
        model,
        tensor,
        device,
        class_index=args.class_index,
    )

    image_array = (
        np.asarray(image).astype(np.float32) / 255.0
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_array)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(image_array)
    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )

    class_name = "defective" if target == 1 else "normal"

    axes[1].set_title(
        f"Grad-CAM: {class_name}\n"
        f"P(defective)={probabilities[1]:.3f}"
    )
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Grad-CAM saved:", output.resolve())


if __name__ == "__main__":
    main()