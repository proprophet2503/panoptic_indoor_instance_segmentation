"""Download every pretrained checkpoint once, sequentially, before parallel workers start (avoids download races)."""
import os

import common as C


def main():
    cfg = C.load_config()
    for key, hp in cfg["MODELS"].items():
        C.log(f"prefetch {key}")
        if hp["impl"] == "yolo":
            from ultralytics import YOLO, settings
            settings.update({"wandb": False})  # workers log to W&B themselves
            YOLO(hp["weights"])
        elif hp["impl"] == "maskrcnn":
            import torchvision
            torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=False)
        elif hp["impl"] == "mask2former":
            from huggingface_hub import snapshot_download
            snapshot_download(hp["checkpoint"])
        elif hp["impl"] == "rfdetr":
            import rfdetr
            getattr(rfdetr, hp["class"])()
    C.log(f"prefetch done (cwd={os.getcwd()})")


if __name__ == "__main__":
    main()
