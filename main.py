import torch

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)
import data_Preprocess
import numpy as np
np.random.seed(0)
import utils
import pandas as pd

def main():
    args, unknown = utils.parse_args()

    if args.embedder == 'WGAN':
        from models import WGAN_ModelTrainer
        embedder = WGAN_ModelTrainer(args)

    embedder.train()
    embedder.writer.close()



if __name__ == "__main__":
    main()
    #imputation("./model_checkpoints/embeddings_Adam_clustering.pt")

