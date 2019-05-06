import argparse
from src.entity_extractor import EntityExtractor
from src.doc_encoder import stackpassage, train, load_model, eval
from src.utils import load_data
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./src/model")
parser.add_argument("--num_epoch",default=16,type=int)
parser.add_argument("--lr", default=1e-2)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--num_candidate",default=10,type=int)
parser.add_argument("--batch_size",default=16,type=int)

arg = parser.parse_args()
MODEL_PATH = arg.model_path
NUM_EPOCH = arg.num_epoch
LEARNING_RATE = arg.lr
EVAL_MODE = arg.eval
NUM_CANDIDATE = arg.num_candidate
DIMENSION = 768
BATCH_SIZE = arg.batch_size

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("using device",device)
    extractor = EntityExtractor(NUM_CANDIDATE)
    data = load_data('./src/data/test.txt')
    data = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
    model, cur_epoch = load_model(MODEL_PATH, hidden_size=DIMENSION)
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if EVAL_MODE:
        print("eval start: ")
        for idx, items in enumerate(data):
            if idx < 1030:
                continue
            if idx > 1040:
                return
            print("start eval batch {}: ".format(idx))
            passages = extractor.get_passages(items, device=device)
            dataloader = stackpassage(passages)
            f1_score = eval(dataloader,model)
            print(f1_score)

    else:
        print("start training")
        for idx, items in enumerate(data):
            if idx < cur_epoch:
                continue
            print("start batch {}: ".format(idx))
            passages = extractor.get_passages(items, device=device)
            dataloader = stackpassage(passages)
            train(dataloader,model,optimizer,device=device,save_path=MODEL_PATH, cur_epoch=cur_epoch)
            cur_epoch += 1
            if cur_epoch == 1030:
                cur_epoch = 0


if __name__ == "__main__":
    main()
    # extractor = EntityExtractor(2)
    # data = load_data('./src/data/test.txt')
    # passages = extractor.get_passages(data)
    # print(passages)