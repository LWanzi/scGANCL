from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.random.seed(0)
import sys
from torch import optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
from utils import EMA, set_requires_grad, init_weights, update_moving_average, loss_fn, repeat_1d_tensor, currentTime, \
    set_grad
import copy
from data import Dataset
from embedder import Discriminator, ZINB_Encoder, Generator
from utils import config2string
import faiss
from ZINB_loss import ZINB, NB
import utils
from data_Preprocess import impute_dropout
from ContraD import supcon_fake, nt_xent
from embedder import embedder

from copy import deepcopy

args_transformation = {
    # mask
    'mask_percentage': 0.7,
    'apply_mask_prob': 0.5,
    'noise_percentage': 0.8,
    'sigma': 0.5,
    'apply_noise_prob': 0.5,
    'swap_percentage': 0.1,
    'apply_swap_prob': 0.5,
    'cross_percentage': 0.25,
    'apply_cross_prob': 0.5,
}


def RandomMask(sample, args_transformation):
    dataset_for_transform = deepcopy(sample)
    # tr = transformation(dataset_for_transform.iloc[idx, :], sample1)
    tr = transformation(dataset_for_transform, sample)  # dataset_for_transform.iloc[:, idx]
    tr.random_mask(args_transformation['mask_percentage'], args_transformation['apply_mask_prob'])
    return tr.cell_profile


def RandomSwap(sample, args_transformation):
    # tr = transformation(self.dataset_for_transform.iloc[:, index], sample)
    dataset_for_transform = deepcopy(sample)
    tr = transformation(dataset_for_transform, sample)
    # # inner swap
    # tr.random_swap(self.args_transformation['swap_percentage'], self.args_transformation['apply_swap_prob'])
    tr.random_swap(args_transformation['swap_percentage'], args_transformation['apply_swap_prob'])
    return tr.cell_profile


def InstanceCrossover(sample, args_transformation):
    # tr = transformation(self.dataset_for_transform.iloc[:, index], sample)
    dataset_for_transform = deepcopy(sample)
    tr = transformation(dataset_for_transform, sample)
    # # inner swap
    # tr.random_swap(self.args_transformation['swap_percentage'], self.args_transformation['apply_swap_prob'])
    tr.instance_crossover(args_transformation['cross_percentage'], args_transformation['apply_cross_prob'])
    return tr.cell_profile


class transformation():
    def __init__(self,
                 dataset,
                 cell_profile):
        self.dataset = dataset
        # self.sct_profile = sct_profile
        # self.cell_profile = torch.empty_like(cell_profile).copy_(cell_profile)
        # c=self.cell_profile
        # self.cell_profile=deepcopy(c)
        self.cell_profile = deepcopy(cell_profile)
        # self.cell_profile = cell_profile.clone()#修改deepcopy，使用deepcopy复制，使叶子节点可以参与复制
        # self.gene_num = len(self.cell_profile)
        self.cell_num = self.cell_profile.shape[0]
        self.gene_num = self.cell_profile.shape[1]
        # self.cell_num = len(self.dataset)
        # self.cell_num = self.sct_profile.shape[1]

    def build_mask(self, masked_percentage: float):
        mask = np.concatenate([np.ones(int(self.gene_num * masked_percentage), dtype=bool),
                               np.zeros(self.gene_num - int(self.gene_num * masked_percentage), dtype=bool)])
        # print(self.gene_num)
        # print(np.ones(int(self.gene_num * masked_percentage)))
        # print(np.zeros(self.gene_num - int(self.gene_num * masked_percentage)))
        np.random.shuffle(mask)
        return mask

    def random_mask(self,
                    mask_percentage: float = 0.7,
                    apply_mask_prob: float = 0.5):
        # s = np.random.uniform(0, 1)
        # if s < apply_mask_prob:
        for i in range(self.cell_profile.shape[0]):
            mask = self.build_mask(mask_percentage)
            self.cell_profile[i][mask] = 0

                # o=self.cell_profile
                # mask = self.build_mask(mask_percentage)
                # b = self.cell_profile.shape[0]  # 得到batch size大小
                # mask = np.tile(mask, (b, 1))  # 复制mask矩阵，得到batch size*gene number的矩阵，对第一维复制b次，对第二维复制1次
                # self.cell_profile[mask] = 0

    def random_swap(self,
                    swap_percentage: float = 0.1,
                    apply_swap_prob: float = 0.5):

        ##### for debug
        #     from copy import deepcopy
        #     before_swap = deepcopy(cell_profile)
        # s = np.random.uniform(0, 1)
        # if s < apply_swap_prob:
            # create the number of pairs for swapping
        swap_instances = int(self.gene_num * swap_percentage / 2)  # 一个细胞中需要交换基因表达值的次数
        swap_pair = np.random.randint(self.gene_num,
                                      size=(
                                          swap_instances, 2))  # 生成随机值从0到gene_num，大小为(swap_instances, 2),生成需要交换基因表达值的对

        # do the inner crossover with p
        # s=self.cell_profile[:,swap_pair[:, 0]]
        # q=self.cell_profile[:,swap_pair[:, 1]]
        self.cell_profile[:, swap_pair[:, 0]], self.cell_profile[:, swap_pair[:, 1]] = \
            self.cell_profile[:, swap_pair[:, 1]], self.cell_profile[:, swap_pair[:, 0]]  # 将一个细胞中需要交换基因表达值的基因进行交换
        # d = self.cell_profile


            # return d

    def instance_crossover(self,
                           cross_percentage: float = 0.25,
                           apply_cross_prob: float = 0.4):

        # it's better to choose a similar profile to crossover

        # s = np.random.uniform(0, 1)
        # if s < apply_cross_prob:
        for i in range(self.cell_num):
            # choose one instance for crossover
            cross_idx = np.random.randint(self.cell_num)  # 随机选择一个细胞
            cross_instance = self.dataset[cross_idx]  # 得到随机选择细胞的基因表达值
            # build the mask
            mask = self.build_mask(cross_percentage)  # 确定需要交换的基因位置True
            # apply instance crossover with p
            tmp = cross_instance[mask].copy()  # 得到随机选择细胞交换后的基因表达值
            cross_instance[mask], self.cell_profile[i][mask] = self.cell_profile[i][mask], tmp

            # print("sr")



class WGAN_ModelTrainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))

    def _init(self):
        args = self._args
        self._task = args.task
        print("Downstream Task : {}".format(self._task))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        # self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        self._device = "cpu"
        # torch.cuda.set_device(self._device)
        self._dataset = Dataset(root=args.root, dataset=args.dataset)
        self._loader = DataLoader(dataset=self._dataset)
        # 设置输入维度为[500,1024]
        layers = [self._dataset.data.x.shape[1]] + self.hidden_layers
        self._model = WGAN_Discriminator(layers, args).to(self._device)
        self.generator = Generator().to(self._device)
        self.generator.apply(init_weights)

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay=1e-5)
        self._optimizer_G = optim.AdamW(params=self.generator.parameters(), lr=args.lr, weight_decay=1e-5)

    def train(self):

        self.best_test_acc, self.best_dev_acc, self.best_test_std, self.best_dev_std, self.best_epoch = 0, 0, 0, 0, 0
        self.best_dev_accs = []

        # get Random Initial accuracy
        self.infer_embeddings(0)
        print("initial accuracy ")
        self.evaluate(self._task, 0)

        f_final = open("results/{}.txt".format(self._args.embedder), "a")

        # Start Model Training
        print("Training Start!")
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):
                # augmentation = utils.Augmentation(float(self._args.aug_params[0]), float(self._args.aug_params[1]),
                #                                   float(self._args.aug_params[2]), float(self._args.aug_params[3]))
                batch_data.to(self._device)
                # view1, view2 = augmentation._feature_masking(batch_data, self._device)
                # D_real_loss,zinb_loss,sim_loss
                emb, loss, sup_f1, sup_f2, D, t_D, pro_s, D_real_loss, zinb_loss, zs, sim_loss, ss = self._model(
                    x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                    neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                    edge_weight=batch_data.edge_attr, epoch=epoch)
                z = Variable(torch.Tensor(np.random.normal(0, 1, (batch_data.x.shape[0],
                                                                  2048)))).to(self._device)
                # d=self.generator(z).detach().cpu().numpy()
                # d2=self.generator(z)
                g_z_f1 = RandomMask(self.generator(z).detach().cpu().numpy(), args_transformation)
                g_z_f2 = RandomSwap(g_z_f1, args_transformation)
                g_z_f = InstanceCrossover(g_z_f2, args_transformation)

                # g_z_f3 = RandomMask(self.generator(z).detach().cpu().numpy(), args_transformation)
                # g_z_f = InstanceCrossover(self.generator(z).detach().cpu().numpy(),args_transformation)
                # g_z_f3 = InstanceCrossover(self.generator(z).detach().cpu().numpy(), args_transformation)
                # g_z_f2 = RandomMask(d2, args_transformation)
                sup_f = pro_s(D(torch.tensor(g_z_f).to(self._device)))
                # sup_f3 = pro_s(D(torch.tensor(g_z_f3).to(self._device)))
                sup_f = F.normalize(sup_f)
                loss_s = supcon_fake(sup_f1, sup_f2, sup_f, 0.1)
                # loss_s = supcon_fake(pro_r1, pro_f, pro_f3, 0.1)
                x_ = self.generator(z)
                # xv=RandomMask(self.generator(z),args_transformation)
                cat_x_ = torch.cat((x_, z), 1)
                # cat_x_ = torch.cat((torch.tensor(g_z_f).to(self._device), z), 1)
                tiny_x_ = t_D(cat_x_)
                loss_fake = -torch.mean(tiny_x_)
                self.generator.requires_grad = False
                loss_a = loss+1e-1*loss_s- loss_fake#1e-2*loss_s*5
                #loss_a = loss + loss_s - loss_fake
                ls =1e-1*loss_s
                self._optimizer.zero_grad()
                self.generator.requires_grad = True
                loss_g = loss_fake
                self._optimizer_G.zero_grad()
                loss_a.backward(retain_graph=True)
                loss_g.backward()
                self._optimizer.step()
                self._optimizer_G.step()
                # emb, loss = self._model(x=view1.x, x2=view2.x, y=batch_data.y, edge_index=view1.edge_index,
                #                         edge_index_2=view2.edge_index,
                #                         neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                #                         edge_weight=view1.edge_attr, edge_weight_2=view2.edge_attr, epoch=epoch)



                # st = '[{}][Epoch {}/{}] Loss g: {:.4f},D_real_loss: {:.4f},zinb_loss: {:.4f},sim_loss: {:.4f},loss_s: {:.4f}'.format(currentTime(), epoch, self._args.epochs,loss_g.item(), D_real_loss.item(),zinb_loss.item(),sim_loss.item(),loss_s.item())#loss.item(),
                st = '[{}][Epoch {}/{}] Loss g: {:.4f},loss_a: {:.4f},D_real_loss: {:.4f},loss_fake: {:.4f},zinb_loss: {:.4f},zs: {:.4f},sim_loss: {:.4f},ss: {:.4f},loss_s: {:.4f},ls: {:.4f}'.format(
                    currentTime(), epoch, self._args.epochs, loss_g.item(), loss_a.item(), D_real_loss.item(),
                    loss_fake.item(), zinb_loss.item(), zs.item(), sim_loss.item(), ss.item(), loss_s.item(),
                    ls.item())  # loss.item()
                # st = '[{}][Epoch {}/{}] Loss g: {:.4f},loss_a: {:.4f},D_real_loss: {:.4f},loss_fake: {:.4f},zinb_loss: {:.4f},zs: {:.4f},sim_loss: {:.4f},ss: {:.4f}'.format(currentTime(), epoch, self._args.epochs, loss_g.item(), loss_a.item(), D_real_loss.item(),loss_fake.item(), zinb_loss.item(),zs.item(), sim_loss.item(), ss.item())  # loss.item()
                print(st)

            if (epoch) % 5 == 0:
                self.infer_embeddings(epoch)
                self.evaluate(self._task, epoch)

        print("\nTraining Done!")
        print("[Final] {}".format(self.st_best))

        f_final.write("{} -> {}\n".format(self.config_str, self.st_best))


class WGAN_Discriminator(nn.Module):
    def __init__(self, layer_config, args, **kwargs):
        super().__init__()
        # student_encoder将输入的数据进行GCN操作
        # image_size=(32,32,1)
        self.discriminator = Discriminator()
        self.zinb_Encoder = ZINB_Encoder()
        self.zinb_Encoder.apply(init_weights)
        # teacher_encoder对student_encoder进行深拷贝
        self.discriminator.apply(init_weights)

        self.relu = nn.ReLU()
        self.topk = args.topk
        # self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        # torch.cuda.set_device(self._device)
        self._device = "cpu"
        self.d_hidden = 128
        self.tinyDiscriminator = nn.Sequential(
            nn.Linear(4096, 512),  # 512
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.d_hidden),  # 512
            nn.LeakyReLU(0.2),
            nn.Linear(self.d_hidden, 1),
            nn.Sigmoid()
        )
        self.projection = nn.Sequential(
            nn.Linear(2048, self.d_hidden),  # d_penul 32768
            nn.LeakyReLU(0.2),
            nn.Linear(self.d_hidden, 64),
            nn.Sigmoid()
        )
        self.projection2 = nn.Sequential(
            nn.Linear(2048, self.d_hidden),  # d_penul 32768
            nn.LeakyReLU(0.2),
            nn.Linear(self.d_hidden, 64),
            nn.Sigmoid()
        )

    def forward(self, x, y, edge_index, neighbor, edge_weight=None, epoch=None):
        # student得到卷积之后的hi
        student = self.discriminator(x)  # discriminator为编码器
        # student_ = self.student_encoder(x=x2, edge_index=edge_index_2, edge_weight=edge_weight_2)

        x_o_np = x.detach().cpu().numpy()
        # x_f1_np = RandomMask(x_o_np,args_transformation)
        # x_f2_np = RandomMask(x_o_np,args_transformation)
        x_f1_np1 = RandomMask(x_o_np, args_transformation)
        x_f1_np2 = RandomSwap(x_f1_np1, args_transformation)
        x_f1_np = InstanceCrossover(x_f1_np2, args_transformation)

        x_f2_np1 = RandomMask(x_o_np, args_transformation)
        x_f2_np2 = RandomSwap(x_f2_np1, args_transformation)
        x_f2_np = InstanceCrossover(x_f2_np2, args_transformation)
        # x_f1_np = InstanceCrossover(x_o_np, args_transformation)
        # x_f2_np = InstanceCrossover(x_o_np, args_transformation)
        x_f1 = torch.tensor(x_f1_np).to(self._device)
        x_f2 = torch.tensor(x_f2_np).to(self._device)
        f1_r = self.discriminator(x_f1)
        f2_r = self.discriminator(x_f2)
        # pro_f1和pro_f2分别是输入的两个增强表示
        pro_f1 = self.projection(f1_r)  # 通过第一个映射头，计算simclr损失
        pro_f2 = self.projection(f2_r)
        pro_f1 = F.normalize(pro_f1)
        pro_f2 = F.normalize(pro_f2)
        # tiny_x代表最终判别器辨别真假
        cat_student = torch.cat((x, student), 1)
        # cat_student = torch.cat((x, f1_r), 1)
        tiny_x = self.tinyDiscriminator(cat_student)  # tinyDiscriminator为鉴别器
        sim_loss = nt_xent(pro_f1, pro_f2)
        sup_f1 = self.projection2(f1_r)  # 通过第二个映射头，计算sup损失
        sup_f2 = self.projection2(f2_r)
        sup_f1 = F.normalize(sup_f1)
        sup_f2 = F.normalize(sup_f2)
        zinb_loss, latent = self.zinb_Encoder(student, x)
        D_real_loss = -torch.mean(tiny_x)
        # pred得到映射器映射出的融合信息z
        # pred = self.student_predictor(student)
        # pred_ = self.student_predictor(student_)
        #loss_all = D_real_loss + sim_loss + zinb_loss  #
        loss_all = D_real_loss+1e-1*sim_loss+1e-6*zinb_loss#1e-2*sim_loss*5
        zs =1e-6*zinb_loss
        ss = 1e-1*sim_loss
        # ind,k返回值暂时去除
        return student, loss_all, sup_f1, sup_f2, self.discriminator, self.tinyDiscriminator, self.projection2, D_real_loss, zinb_loss, zs, sim_loss, ss  #