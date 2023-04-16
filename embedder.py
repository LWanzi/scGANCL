import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
from models import LogisticRegression
from utils import printConfig
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import utils
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)
import data_Preprocess
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise,adjusted_rand_score
from ZINB_loss import ZINB, NB
from torch.autograd import Variable


class embedder:
    def __init__(self, args):
        self.args = args
        self.hidden_layers = eval(args.layers)
        printConfig(args)

    def infer_embeddings(self, epoch):
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._dev_mask = self._test_mask = None
        for bc, batch_data in enumerate(self._loader):
            # augmentation = utils.Augmentation(float(self._args.aug_params[0]), float(self._args.aug_params[1]),
            #                                   float(self._args.aug_params[2]), float(self._args.aug_params[3]))

            batch_data.to(self._device)
            # view1, view2 = augmentation._feature_masking(batch_data, self._device)
#D_real_loss,zinb_loss,sim_loss
            emb, loss, pro_r1, pro_r2, D, t_D,_,D_real_loss,zinb_loss,zs,sim_loss,ss= self._model(x = batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                                                           neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                                                           edge_weight=batch_data.edge_attr, epoch=epoch)
            d_z = self.generator(emb)
            print(d_z.shape)
            # emb, loss = self._model(x=view1.x, x2=view2.x, y=batch_data.y, edge_index=view1.edge_index,
            #                         edge_index_2=view2.edge_index,
            #                         neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
            #                         edge_weight=view1.edge_attr, edge_weight_2=view2.edge_attr, epoch=epoch)
            emb = d_z.detach()
            y = batch_data.y.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])

    def evaluate(self, task, epoch):
        if task == "node":
            self.evaluate_node(epoch)
        elif task == "clustering":
            self.evaluate_clustering(epoch)
        elif task == "similarity":
            self.run_similarity_search(epoch)
        

    def evaluate_node(self, epoch):

        # print()
        # print("Evaluating ...")
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]

        dev_accs, test_accs = [], []

        for i in range(20):

            self._train_mask = self._dataset[0].train_mask[i]
            self._dev_mask = self._dataset[0].val_mask[i]
            if self._args.dataset == "wikics":
                self._test_mask = self._dataset[0].test_mask
            else:
                self._test_mask = self._dataset[0].test_mask[i]

            classifier = LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

            for _ in range(100):
                classifier.train()
                logits, loss = classifier(self._embeddings[self._train_mask], self._labels[self._train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            dev_logits, _ = classifier(self._embeddings[self._dev_mask], self._labels[self._dev_mask])
            test_logits, _ = classifier(self._embeddings[self._test_mask], self._labels[self._test_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() /
                       self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() /
                        self._labels[self._test_mask].shape[0]).detach().cpu().numpy()

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        print('** [{}] [Epoch: {}] Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(self.args.embedder, epoch, dev_acc, dev_std, test_acc, test_std))

        if dev_acc > self.best_dev_acc:
            self.best_dev_acc = dev_acc
            self.best_test_acc = test_acc
            self.best_dev_std = dev_std
            self.best_test_std = test_std
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best val | Best test: {:.4f} ({:.4f}) / {:.4f} ({:.4f})**\n'.format(
            self.best_epoch, self.best_dev_acc, self.best_dev_std, self.best_test_acc, self.best_test_std)
        print(self.st_best)


    def evaluate_clustering(self, epoch):
        
        embeddings = F.normalize(self._embeddings, dim = -1, p = 2).detach().cpu().numpy()
        print(embeddings.shape)
        nb_class = len(self._dataset[0].y.unique())
        true_y = self._dataset[0].y.detach().cpu().numpy()

        estimator = KMeans(n_clusters = nb_class)

        NMI_list = []
        ARI_list=[]
        self.aris = []
        for i in range(10):
            estimator.fit(embeddings)
            y_pred = estimator.predict(embeddings)

            s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
            s2 = adjusted_rand_score(true_y, y_pred)
            NMI_list.append(s1)
            ARI_list.append(s2)


        s1 = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)

        #用于生成轨迹推断数据集的插补，以余弦相似性为指标
        # data_original = pd.read_csv("./test_csv/Klein/process_data.csv", header=None, sep=",")
        # X_original = np.array(data_original)[1:, 1:].T
        # count_for_original = torch.tensor(X_original)
        # count_for_original_ = F.normalize(count_for_original, dim=-1, p=2).detach().cpu().numpy()
        # X_zero, i, j, ix = data_Preprocess.impute_dropout(X_original, 1, 0.1)
        # mean, median, min, max = data_Preprocess.imputation_error(embeddings, count_for_original_, X_zero, i, j, ix)
        # cosine_sim = data_Preprocess.imputation_cosine(self._embeddings.detach().cpu().numpy(), X_original, X_zero, i, j, ix)
        # s1=cosine_sim

        ##recover ability
        # data_original = pd.read_csv("./results/Klein/raw-Klein-imputed.csv", header=None, sep=",")
        # X_original = np.array(data_original)[1:, 1:].T
        # count_for_original = torch.tensor(X_original)
        # count_for_original_ = F.normalize(count_for_original, dim=-1, p=2).detach().cpu().numpy()
        # X_zero, i, j, ix = data_Preprocess.impute_dropout(X_original, 1, 0.1)
        # mean, median, min, max = data_Preprocess.imputation_error(embeddings, count_for_original_, X_zero, i, j, ix)
        # cosine_sim = data_Preprocess.imputation_cosine(self._embeddings.detach().cpu().numpy(), X_original, X_zero, i, j, ix)
        print('** [{}] [Current Epoch {}] Clustering NMI: {:.4f},Clustering ARI: {:.4f} **'.format(self.args.embedder,epoch, s1, s2))

        if s1 > self.best_dev_acc:
            self.best_epoch = epoch
            self.best_dev_acc = s1
            self.ari = s2
            print("~~~~~~~~~~~~~~~~~~")
            print(self.best_dev_acc)
            if self._args.checkpoint_dir is not '':
                print('Saving checkpoint...')
                torch.save(embeddings, os.path.join(self._args.checkpoint_dir, 'embeddings_{}_{}.pt'.format(self._args.dataset, self._args.task)))
                # zzz = np.concatenate((true_y.reshape(3660, 1), y_pred.reshape(3660, 1)), axis=1)
                a = pd.DataFrame(self._embeddings.detach().cpu().numpy()).T
                a.to_csv("./results/student.csv")
            print("save")
            print("~~~~~~~~~~~~~~~~~~")

        self.best_dev_accs.append(self.best_dev_acc)
        self.aris.append(self.ari)
        # self.st_best = '** [Best epoch: {}] Best NMI: {:.4f} **\n'.format(self.best_epoch, self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best NMI: {:.4f}  Current ARI: {:.4f} **\n'.format(self.best_epoch,self.best_dev_acc,self.ari)
        # self.st_best = '** [Best epoch: {}] Best Cos Similarity: {:.4f} **\n'.format(self.best_epoch, self.best_dev_acc)
        print(self.st_best)



    def run_similarity_search(self, epoch):

        test_embs = self._embeddings.detach().cpu().numpy()
        test_lbls = self._dataset[0].y.detach().cpu().numpy()
        numRows = test_embs.shape[0]

        cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
        st = []
        for N in [5, 10]:
            indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
            tmp = np.tile(test_lbls, (numRows, 1))
            selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
            original_label = np.repeat(test_lbls, N).reshape(numRows,N)
            st.append(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4))

        print("** [{}] [Current Epoch {}] sim@5 : {} | sim@10 : {} **".format(self.args.embedder, epoch, st[0], st[1]))

        if st[0] > self.best_dev_acc:
            self.best_dev_acc = st[0]
            self.best_test_acc = st[1]
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best @5 : {} | Best @10: {} **\n'.format(self.best_epoch, self.best_dev_acc, self.best_test_acc)
        print(self.st_best)

        return st


class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=2048, input_size=48):
    # def __init__(self, input_dim=1, output_dim=512, input_size=4):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        layer_config = [2048,2048]
        self.stacked_en = nn.ModuleList(
            [nn.Linear(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList(
            [nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.Sigmoid() for _ in range(1, len(layer_config))])

    def forward(self, x):
        for i, gnn in enumerate(self.stacked_en):
            # x = gnn(x, edge_index, edge_weight=edge_weight)
            x = gnn(x)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)

        return x



class ZINB_Encoder(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, rep_dim=256):
        super(ZINB_Encoder, self).__init__()
        dec_dim = [512, 256]
        # dec_dim = [512, 32]  # [512, 256]
        rep_dim = 2048
        self.ZINB_Encoder = nn.Sequential(nn.Linear(rep_dim, dec_dim[0]), nn.PReLU(),
                                          nn.Linear(dec_dim[0], dec_dim[1]), nn.PReLU())
        self.pi_Encoder = nn.Sequential(nn.Linear(dec_dim[1], 2048), nn.Sigmoid())
        self.disp_Encoder = nn.Sequential(nn.Linear(dec_dim[1], 2048), nn.Softplus())
        self.mean_Encoder = nn.Linear(dec_dim[1], 2048)

    def clip_by_tensor(self, t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """

        # t = torch.tensor(t, dtype=torch.float32)
        t = t.clone().detach()
        t_min = torch.tensor(t_min, dtype=torch.float32)
        t_max = torch.tensor(t_max, dtype=torch.float32)
        #
        # result = torch.tensor((t >= t_min), dtype=torch.float32) * t + torch.tensor((t < t_min),
        #                                                                             dtype=torch.float32) * t_min
        # result = torch.tensor((result <= t_max), dtype=torch.float32) * result + torch.tensor((result > t_max),
        #                                                                                       dtype=torch.float32) * t_max

        # t_min = t_min.clone().detach()
        # t_max = t_max.clone().detach()
        result = ((t >= t_min).clone().detach())* t + ((t <= t_min).clone().detach())* t_min
        result = ((result <= t_max).clone().detach()) * result + ((result > t_max).clone().detach()) * t_max
        return result

    def forward(self, input,x):
        latent_z = self.ZINB_Encoder(input)
        pi = self.pi_Encoder(latent_z)
        disp = self.disp_Encoder(latent_z)
        disp = self.clip_by_tensor(disp, 1e-4, 1e4)
        mean = self.mean_Encoder(latent_z)
        mean = self.clip_by_tensor(torch.exp(mean), 1e-5, 1e6)
        zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
        zinb_loss = zinb.loss(x, mean, mean=True)
        return zinb_loss, latent_z




class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=48):
    # def __init__(self, input_dim=10, output_dim=1, input_size=5):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()#ReLU  Softplus Sigmoid
        )

    def forward(self, input):
        x = self.fc(input)

        return x