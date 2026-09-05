import torch
import torch.nn as nn
import random
import numpy as np
from model_VAE_new import VAE
def gaussian_reparameterization_std(means, std, times=1):
    std = std.abs()
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var+1e-8)
    assert torch.sum(std<0).item()==0
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        self.mlps =[]
        layers = []

        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
        return x

class Qc_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(Qc_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())

    def forward(self, x):
        assert torch.sum(torch.isnan(x)).item() == 0
        hidden_features = self.mlp(x)
        c_mu = self.z_loc(hidden_features)
        c_sca = self.z_sca(hidden_features)
        assert torch.sum(torch.isinf(c_mu)).item() == 0
        return c_mu, c_sca

class Net(nn.Module):
    def __init__(self, d_list,num_classes,z_dim,adj,rand_seed=0):
        super(Net, self).__init__()
        self.rand_seed = rand_seed

        self.label_embedding_u = nn.Parameter(torch.eye(num_classes),
                                            requires_grad=True)
        self.label_embedding_std = nn.Parameter(torch.ones(num_classes),
                                            requires_grad=True)
        self.label_adj = nn.Parameter(torch.eye(num_classes),
                                      requires_grad=True)
        self.register_buffer('adj', adj)
        self.z_dim = z_dim
        self.label_mlp = Qc_inference_mlp(num_classes, z_dim)
        self.mix_prior = None
        self.mix_mu = None
        self.mix_sca = None
        self.k = num_classes
        self.VAE = VAE(d_list=d_list,z_dim=z_dim,class_num=num_classes)
        self.cls_conv = nn.Conv1d(num_classes, num_classes,
                                  z_dim*2, groups=num_classes)

        self.cls = nn.Linear(z_dim, num_classes)
        self.view_cls = nn.ModuleList([nn.Linear(z_dim, num_classes) for i in range(len(d_list)) ])
        self.set_prior()
    def set_prior(self):
        self.mix_prior = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        self.mix_mu = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
        self.mix_sca = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)

    def poe_two(self, z_mu, z_var, c_mu, c_var, eps=1e-5):

        z_mu = z_mu.unsqueeze(1)
        z_var = z_var.unsqueeze(1)
        c_mu = c_mu.unsqueeze(0)
        c_var = c_var.unsqueeze(0)


        s_mu = z_mu.new_zeros([z_mu.shape[0], c_mu.shape[1], z_mu.shape[2]])
        s_var = z_var.new_ones([z_var.shape[0], c_var.shape[1], z_var.shape[2]])
        T_x = 1. / (z_var+eps)
        T_c = 1. / (c_var+eps)
        T_s = 1. / (s_var+eps)



        T_sum = T_x + T_c + T_s
        aggregate_mu = (z_mu*T_x+c_mu*T_c+ s_mu*T_s)/T_sum

        aggregate_var = 1. / T_sum
        assert torch.sum(torch.isnan(aggregate_mu)).item()==0
        assert torch.sum(torch.isinf(aggregate_mu)).item()==0

        assert torch.sum(torch.isnan(aggregate_var)).item()==0
        assert torch.sum(torch.isinf(aggregate_var)).item()==0
        return aggregate_mu, aggregate_var

    def forward(self, x_list, row_index, mask):
        label_embedding  =  self.label_embedding_u
        label_embedding_sample = self.label_embedding_u
        label_embedding_var = self.label_embedding_u
        prop = 0.25
        model_input = x_list
        if self.training:
            model_input = []
            for X in x_list:
                bit_mask_len = int(prop*X.size(-1))
                bit_mask = torch.ones_like(X)
                for j in range(mask.shape[0]):
                    zero_indices = torch.randperm(bit_mask.shape[1], device=X.device)[:bit_mask_len]
                    bit_mask[j, zero_indices] = 0
                model_input.append(X * bit_mask)


        if torch.sum(torch.isnan(label_embedding)).item() > 0:
            assert torch.sum(torch.isnan(label_embedding)).item() == 0
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, cross_enc_mu, cross_enc_var = self.VAE(model_input, row_index, mask)
        pred = self.cls(z_sample[0])
        pred_perm = self.cls(z_sample[1])

        vis_sampling = gaussian_reparameterization_var(cross_enc_mu[0].diagonal(dim1=1, dim2=2),cross_enc_var[0].diagonal(dim1=1, dim2=2)).transpose(1,2)
        pred_diag = self.cls(vis_sampling)

        pred = torch.sigmoid(pred)
        pred_perm = torch.sigmoid(pred_perm)

        return z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, [pred, pred_perm], label_embedding_sample, label_embedding, label_embedding_var, cross_enc_mu, cross_enc_var,pred_diag

def get_model(d_list,num_classes,z_dim,adj,rand_seed=0):


    model = Net(d_list,num_classes=num_classes,z_dim=z_dim,adj=adj,rand_seed=rand_seed)
    model = model.to(torch.device('cuda' if torch.cuda.is_available()
                                    else 'cpu'))
    return model
