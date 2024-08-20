BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
EN_VOCAB_SIZE = 52000  # 1000
DE_VOCAB_SIZE = 52000  # 1000

TRAIN_GPU = 0
NUM_EPOCHS = 10
BASE_LEARNING_RATE = 1.0
BATCH_SIZE = 40
ACCUM_ITER = 10
MAX_PADDING = 72
WARMUP = 3000
FILE_PREFIX = "multi30k_model_"
MAX_SEQ_LEN = 5000  # 50
VALIDATION_SAMPLE_SIZE = 100

D_MODEL = 512
N_LAYERS = 6  # 6
N_HEADS = 8  # 8
DROPOUT = 0.1