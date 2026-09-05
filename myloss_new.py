import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_div_var(q_mu, q_var, p_mu, p_var, eps=1e-12):
    return 0.5 * (torch.log((p_var+eps) / (q_var+eps)) + (q_var+eps) / (p_var+eps) + torch.pow(q_mu - p_mu, 2) / (p_var+eps) - 1)


class Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super(Loss, self).__init__()
        self.temperature = temperature

    def uni_comb_corh_loss(self,cross_enc_mu,cross_enc_var,mu_comb_views,var_comb_views,mask):
        #cross_enc_mu [b m m d]
        #mu_comb_views [m b d]
        #mask [b m]
        num_views = mu_comb_views.shape[0]
        batch_size = mu_comb_views.shape[1]
        dim = mu_comb_views.shape[2]
        mu0 = cross_enc_mu.view(-1,cross_enc_mu.shape[-1])
        var0 = cross_enc_var.view(-1,cross_enc_mu.shape[-1])

        mu1 = mu_comb_views.permute(1,0,2).unsqueeze(2).repeat(1,1,num_views,1).view(-1,mu_comb_views.shape[-1])
        var1 = var_comb_views.permute(1,0,2).unsqueeze(2).repeat(1,1,num_views,1).view(-1,var_comb_views.shape[-1])
        KL = kl_div_var(mu1,var1,mu0,var0).view(batch_size, num_views, num_views, dim) #[b m m d]
        KL = torch.mean(KL,dim=(2,3))
        KL = KL * mask
        return KL.sum() / mask.sum().clamp_min(1)
    def contrastive_loss(self, input_mu, mask_v, temperature=None):
        if temperature is None:
            temperature = self.temperature
        shared_mu = torch.transpose(input_mu,0,1)
        _, n_view, _ = shared_mu.shape
        total_loss = 0
        valid_pairs = 0

        for i in range(n_view):
            for j in range(i + 1, n_view):
                valid_mask = (mask_v[:, i] > 0) & (mask_v[:, j] > 0)

                valid_indices = torch.where(valid_mask)[0]

                if len(valid_indices) < 2:
                    continue

                anchor = shared_mu[valid_indices, i, :]
                positive = shared_mu[valid_indices, j, :]
                anchor = F.normalize(anchor, p=2, dim=1)
                positive = F.normalize(positive, p=2, dim=1)

                all_sim = torch.mm(anchor, positive.t()) / temperature

                pos_sim = torch.diag(all_sim)
                log_sum_exp = torch.logsumexp(all_sim, dim=1)
                losses = -pos_sim + log_sum_exp

                total_loss += torch.sum(losses)

                valid_pairs += len(valid_indices)

        return total_loss / max(valid_pairs, 1) if valid_pairs > 0 else torch.tensor(0.0, device=shared_mu.device)

    def view_specific_perm_KL(self, p_mu, p_var,q_mu, q_var, inc_ind_V):
        z_dim = q_mu.shape[-1]
        mu1, mu0 = q_mu.view(-1, z_dim), p_mu.view(-1, z_dim)
        var1, var0 = q_var.view(-1, z_dim), p_var.view(-1, z_dim)
        KLV = torch.mean(kl_div_var(mu1, var1, mu0, var0), dim=-1)

        KLV = KLV.view(q_mu.shape[0],q_mu.shape[1],q_mu.shape[1]).mean(dim=-1)# [b m]
        mask = inc_ind_V / inc_ind_V.sum(dim=-1, keepdim=True).clamp_min(1)
        KLV = KLV * mask
        return KLV.sum()/mask.shape[0]
    def weighted_BCE_loss(self,pred,label,inc_L_ind,reduction='mean'):
        pred = pred.clamp(min=1e-5, max=1 - 1e-5)
        res = -(label * torch.log(pred) + (1-label) * torch.log(1-pred)) * inc_L_ind
        assert torch.sum(torch.isnan(res)).item() == 0
        assert torch.sum(torch.isinf(res)).item() == 0

        if reduction=='mean':
            return torch.sum(res) / torch.sum(inc_L_ind).clamp_min(1)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res

    def weighted_wmse_loss(self,input, target, weight, reduction='mean'):
        ret = ((target - input) * weight.unsqueeze(-1)) ** 2
        if reduction == 'mean':
            return torch.mean(ret)
        elif reduction=='sum':
            return torch.sum(ret)
        elif reduction=='none':
            return ret
