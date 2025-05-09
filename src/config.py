# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.001
DROPOUT = 0.25

BATCH_SIZE = 32
NUM_WORKERS = 2

# Data Augmentation
ROTATION_RANGE_MIN = 0
ROTATION_RANGE_MAX = 150
FLIPPING = 0.5

BRIGHTNESS = 0.2
CONTRAST = 0.2
SATURATION = 0.2
HUE = 0.1

# Data Normalization
MEAN = [0.5450, 0.4435, 0.3436]
STD = [0.2302, 0.2409, 0.2387]

# dataset specific
NUM_CLASSES = 101

# Data path
PATH = "data"

VALIDATION_SPLIT = 0.2
UNFREEZE_EPOCHS = 5
FINE_TUNE_LR = 1e-4
UNFREEZE_BLOCK = 12