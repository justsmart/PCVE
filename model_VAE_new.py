import torch
import torch.nn as nn

def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var)
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)/(label.sum(dim=1,keepdim=True)+1e-8)
    new_x =  x_embedding*inc_V_ind.T.unsqueeze(-1) + fea.unsqueeze(0)*(1-inc_V_ind.T.unsqueeze(-1))
    return new_x

def batch_rotate_with_prebuilt_index(batch_data, index_row):
    """
    直接使用预构建的 index_row 进行选择。
    参数
    - batch_data: [B, M, M, D]
    - index_row:  [B, M, M]，index_row[b, new_row, col] = old_row
    返回
    - out: [B, M, M, D]
    """
    device = batch_data.device
    B, M, _, D = batch_data.shape
    index_row = index_row.to(device, dtype=torch.long)
    cols = torch.arange(M, device=device, dtype=torch.long)
    index_col = cols.view(1, 1, M).expand(B, M, M)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, M)
    return batch_data[batch_idx, index_row, index_col]



class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,dropout_rate=0.,final_act=True,final_norm=True):
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
class sharedQz_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(sharedQz_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())

    def forward(self, x):
        hidden_features = self.mlp(x)
        z_mu = self.z_loc(hidden_features)
        z_sca = self.z_sca(hidden_features)
        return z_mu, z_sca

class inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)

    def forward(self, x):
        hidden_features = self.mlp(x)
        return hidden_features

class Px_generation_mlp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[512]):
        super(Px_generation_mlp, self).__init__()
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim,final_act=False,final_norm=False)


    def forward(self, z):
        xr = self.mlp(z)
        return xr

class VAE(nn.Module):
    def __init__(self, d_list,z_dim,class_num):
        super(VAE, self).__init__()
        self.x_dim_list = d_list
        self.k = class_num
        self.z_dim = z_dim
        self.num_views = len(d_list)


        self.crossM_enc_matrix = nn.ModuleDict()
        for i in range(self.num_views):
            for j in range(self.num_views):
                self.crossM_enc_matrix["enc%d%d" % (i, j)] = inference_mlp(self.x_dim_list[i],self.z_dim,[])

        self.z_inference = []
        for v in range(self.num_views):
            self.z_inference.append(inference_mlp(self.x_dim_list[v], self.z_dim))
        self.qz_inference = nn.ModuleList(self.z_inference)
        self.qz_inference_header = sharedQz_inference_mlp(self.z_dim, self.z_dim)
        self.x_generation = []
        for v in range(self.num_views):
            self.x_generation.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
        self.px_generation = nn.ModuleList(self.x_generation)
        self.px_generation2 = nn.ModuleList(self.x_generation)


    def inference_z(self, x_list, row_index, mask=None):
        batch_size = x_list[0].shape[0]
        cross_enc_mu = torch.zeros(batch_size, self.num_views, self.num_views, self.z_dim).to(x_list[0].device)
        cross_enc_var = torch.zeros(batch_size, self.num_views, self.num_views, self.z_dim).to(x_list[0].device)

        for i in range(self.num_views):
            for j in range(self.num_views):
                fea = self.crossM_enc_matrix["enc%d%d"%(i,j)](x_list[i])
                cross_enc_mu[:,i,j,:], cross_enc_var[:,i,j,:] = self.qz_inference_header(fea)

        cross_enc_mu_perm = batch_rotate_with_prebuilt_index(cross_enc_mu, row_index)
        cross_enc_var_perm = batch_rotate_with_prebuilt_index(cross_enc_var, row_index)

        mu_comb_views, var_comb_views = self.combination_views(cross_enc_mu,cross_enc_var)
        mu_comb_views_perm, var_comb_views_perm = self.combination_views(cross_enc_mu_perm,cross_enc_var_perm)

        return [mu_comb_views, var_comb_views], [mu_comb_views_perm, var_comb_views_perm], [cross_enc_mu,cross_enc_var],[cross_enc_mu_perm,cross_enc_var_perm]

    def generation_x(self, z):

        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation[v](z)
            xr_dist.append(xrs_loc)
        return xr_dist

    def generation_x_p(self, z):

        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation2[v](z)
            xr_dist.append(xrs_loc)
        return xr_dist

    def combination_views(self, mu, var, eps=1e-5):
        # input [b, m, m ,d]
        zero_1 = mu.new_zeros([1, mu.shape[1], mu.shape[0], mu.shape[3]])
        one_1 = var.new_ones([1, var.shape[1], var.shape[0], var.shape[3]])

        mu_new = torch.cat([zero_1,mu.permute(2,1,0,3)],dim=0)
        var_new = torch.cat([one_1,var.permute(2,1,0,3)],dim=0)


        T = 1. / (var_new+eps)
        aggregate_T = torch.sum(T, dim=0)
        aggregate_var = 1. / (aggregate_T + eps)          # [m b d]
        aggregate_mu = torch.sum(mu_new * T, dim=0) / (aggregate_T + eps)# [m b d]

        return aggregate_mu, aggregate_var

    def poe_aggregate(self, mu, var, mask=None, eps=1e-5):
        # mu, var: [m, b, d]
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        mask_matrix_new = torch.cat([mask_matrix.new_ones([1, mask_matrix.shape[1], mask_matrix.shape[2]]), mask_matrix], dim=0)
        p_z_mu = mu.new_zeros([1, mu.shape[1], mu.shape[2]])
        p_z_var = var.new_ones([1, var.shape[1], var.shape[2]])
        mu_new = torch.cat([p_z_mu,mu],dim=0)
        var_new = torch.cat([p_z_var,var],dim=0)
        exist_mu = mu_new * mask_matrix_new

        T = 1. / (var_new+eps)
        exist_T = T * mask_matrix_new
        aggregate_T = torch.sum(exist_T, dim=0)
        aggregate_var = 1. / (aggregate_T + eps)
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / (aggregate_T + eps)
        return aggregate_mu, aggregate_var

    def moe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        exist_mu = mu * mask_matrix
        exist_var = var * mask_matrix
        aggregate_var = exist_var.sum(dim=0)
        aggregate_mu = exist_mu.sum(dim=0)
        return aggregate_mu,aggregate_var

    def forward(self, x_list, row_index, mask=None):
        [mu_views, var_views], [mu_views_perm, var_views_perm], [cross_enc_mu,cross_enc_var], [cross_enc_mu_perm,cross_enc_var_perm]= self.inference_z(x_list,row_index, mask)


        fusion_mu, fusion_sca = self.poe_aggregate(mu_views, var_views, mask)
        fusion_mu_perm, fusion_sca_perm = self.poe_aggregate(mu_views_perm, var_views_perm, mask)
        assert torch.sum(fusion_sca<0).item() == 0
        z_sample = gaussian_reparameterization_var(fusion_mu, fusion_sca, times=10)
        z_sample_perm = gaussian_reparameterization_var(fusion_mu_perm, fusion_sca_perm, times=10)
        xr_list = self.generation_x(z_sample)
        xr_list_perm = self.generation_x(z_sample_perm)

        return [z_sample,z_sample_perm], [mu_views,mu_views_perm], [var_views,var_views_perm],\
    [fusion_mu,fusion_mu_perm], [fusion_sca,fusion_sca_perm], [xr_list,xr_list_perm], [cross_enc_mu, cross_enc_mu_perm], [cross_enc_var, cross_enc_var_perm]
