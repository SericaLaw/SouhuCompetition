import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./model")
parser.add_argument("--num_epoch",default=10,type=int)
parser.add_argument("--lr", default=1e-2)
parser.add_argument("--eval", action="store_true")

arg = parser.parse_args()
MODEL_PATH = arg.model_path
NUM_EPOCH = arg.num_epoch
LEARNING_RATE = arg.lr
EVAL_MODE = arg.eval
