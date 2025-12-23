import torch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

EMBED_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
MLP_RATIO = 4.0
DROPOUT = 0.1

NUM_CLASSES = 2  # Oxford Pets: background (0) and pet (1)

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 50

DATA_DIR = "./data/oxford_pets"
CHECKPOINT_DIR = "./checkpoints"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
