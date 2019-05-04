import argparse
from src.entity_extractor import EntityExtractor
from src.doc_encoder import stackpassage, PassageEncoder, train, load_model
from src.utils import load_data
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./src/model")
parser.add_argument("--num_epoch",default=10,type=int)
parser.add_argument("--lr", default=1e-2)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--num_candidate",default=10,type=int)
parser.add_argument("--batch_size",default=10,type=int)

arg = parser.parse_args()
MODEL_PATH = arg.model_path
NUM_EPOCH = arg.num_epoch
LEARNING_RATE = arg.lr
EVAL_MODE = arg.eval
NUM_CANDIDATE = arg.num_candidate
DIMENSION = 768
BATCH_SIZE = arg.batch_size

def main():
    extractor = EntityExtractor(NUM_CANDIDATE)
    data = load_data('./src/data/test.txt')
    data = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
    model, cur_epoch = load_model(MODEL_PATH, DIMENSION)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("start training")
    for idx, items in enumerate(data):
        passages = extractor.get_passages(items)
        dataloader = stackpassage(passages)
        train(dataloader,model,optimizer,save_path=MODEL_PATH, cur_epoch=cur_epoch)
        cur_epoch += 1


if __name__ == "__main__":
    main()
