import os
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model_new import get_model
import evaluation
import torch
import numpy as np
import copy
from myloss_new import Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts


def train(loader, model, loss_model, opt, sche, epoch, args, logger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    device = next(model.parameters()).device
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind, row_index) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to(device) for v_data in data]
        label = label.to(device)
        inc_V_ind = inc_V_ind.float().to(device)
        inc_L_ind = inc_L_ind.float().to(device)
        row_index = row_index.to(device)
        z_sample, views_dis_mu, views_dis_var, common_dis_mu, common_dis_var, xr_list,\
        pred, label_emb_sample, label_emb,label_emb_var, cross_enc_mu, cross_enc_var, pred_diag = model(data,row_index,mask=inc_V_ind )
        loss_CL = loss_model.weighted_BCE_loss(pred[0],label,inc_L_ind)

        UC_corh_loss = 0
        UC_corh_loss = loss_model.uni_comb_corh_loss(cross_enc_mu[0],cross_enc_var[0],views_dis_mu[0],views_dis_var[0],inc_V_ind)
        if epoch>=args.pre_epochs:

            UC_corh_loss += loss_model.uni_comb_corh_loss(cross_enc_mu[1],cross_enc_var[1],views_dis_mu[1],views_dis_var[1],inc_V_ind)
            UC_corh_loss = UC_corh_loss / 2

        loss_mse = 0
        for v in range(len(data)):
            loss_mse1 = loss_model.weighted_wmse_loss(xr_list[0][v],data[v],inc_V_ind[:,v],reduction='mean')
            if epoch>=args.pre_epochs:
                loss_mse2 = loss_model.weighted_wmse_loss(xr_list[1][v],data[v],inc_V_ind[:,v],reduction='mean')
                loss_mse += (loss_mse1 + loss_mse2)/2
            else: loss_mse += loss_mse1

        loss_vsp_KL = 0
        loss_vsp_KL = loss_model.view_specific_perm_KL(cross_enc_mu[0],cross_enc_var[0],cross_enc_mu[1],cross_enc_var[1],inc_V_ind)
        if epoch>=args.pre_epochs:
            loss_vsp_KL += loss_model.view_specific_perm_KL(cross_enc_mu[1],cross_enc_var[1],cross_enc_mu[0],cross_enc_var[0],inc_V_ind)
            loss_vsp_KL = loss_vsp_KL / 2

        contras_loss = loss_model.contrastive_loss(views_dis_mu[0], inc_V_ind,temperature=0.1)

        assert torch.sum(torch.isnan(loss_mse)).item() == 0
        loss = loss_CL + loss_mse *args.alpha  + loss_vsp_KL *args.beta + contras_loss*args.gamma + UC_corh_loss * args.sigma
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))

        opt.step()
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()

    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses, model

@torch.no_grad()
def test(loader, model, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    device = next(model.parameters()).device
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind, row_index) in enumerate(loader):
        data=[v_data.to(device) for v_data in data]
        inc_V_ind = inc_V_ind.float().to(device)
        row_index = row_index.to(device)
        z_sample, views_dis_mu, views_dis_var, common_dis_mu, common_dis_var, xr_list, pred, label_emb_sample, label_emb,label_emb_var,_,_,_ = model(data,row_index,mask=inc_V_ind)
        P = pred[0].cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,P.detach().numpy()),axis=0) if len(total_preds)>0 else P.detach().numpy()

        loss=loss_model.weighted_BCE_loss(P,label,inc_L_ind)

        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        losses = losses,
                        ap=evaluation_results[0],
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results

def seed_torch(seed=1029):
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def main(args,file_path):
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    seed_torch(args.seed)
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' +
                                str(args.training_sample_ratio) + '.mat')

    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' +
                                    str(args.mask_label_ratio) + '_T_' +
                                    str(args.training_sample_ratio) + '_'+str(args.alpha)+'_'+str(args.beta)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold_idx in range(folds_num):
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=args.workers)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=args.workers)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        labels = torch.tensor(train_dataset.cur_labels).float().to(device)
        dep_graph = torch.matmul(labels.T,labels)
        dep_graph = dep_graph/(torch.diag(dep_graph).unsqueeze(1)+1e-10)
        dep_graph.fill_diagonal_(fill_value=0.)
        model=get_model(d_list,num_classes=classes_num,z_dim=args.z_dim,adj=dep_graph,rand_seed=0)
        loss_model = Loss()
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = None

        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
        logger.info(str(args))
        static_res = float('-inf')
        epoch_results = [AverageMeter() for _ in range(9)]
        train_curve = []
        best_epoch=0
        best_model_dict = {'model': copy.deepcopy(model.state_dict()), 'epoch': 0}
        for epoch in range(args.epochs):
            train_losses, model = train(train_dataloder, model, loss_model, optimizer, scheduler, epoch, args, logger)
            train_curve.append(train_losses.avg)
            val_results = test(val_dataloder,model,loss_model,epoch,logger)
            for metric, value in zip(epoch_results, val_results):
                metric.update(value)

            selection_score = (val_results[0] + val_results[2] + val_results[3]) / 3
            if selection_score >= static_res:
                static_res = selection_score
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch=epoch
        model.load_state_dict(best_model_dict['model'])
        test_results = test(test_dataloder,model,loss_model,best_epoch,logger)
        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,best_epoch,test_results[0],test_results[1],
            test_results[2],test_results[3]))

        for i in range(9):
            folds_results[i].update(test_results[i])
        # if args.save_curve:
        #     np.save(osp.join(args.curve_dir,args.dataset+'_V_'+str(args.mask_view_ratio)+'_L_'+str(args.mask_label_ratio))+'_'+str(fold_idx)+'.npy', np.array(list(zip(epoch_results[0].vals, train_curve))))
    with open(file_path, mode='a') as file_handle:
        if os.path.getsize(file_path) == 0:
            file_handle.write(
                'AP 1-HL 1-RL AUCme 1-oneE 1-Cov macAUC macro_f1 micro_f1 lr alpha beta gamma sigma\n')
        res_list = [str(round(res.avg,3))+'+'+str(round(res.std,3)) for res in folds_results]
        res_list.extend([str(args.lr),str(args.alpha),str(args.beta),str(args.gamma),str(args.sigma)])
        file_handle.write(' '.join(res_list) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', action='store_true')
    parser.add_argument('--records-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--mask-view-ratio', type=float, default=0.5) # view missing ratio, 0.5 means 50% of the views are missing
    parser.add_argument('--mask-label-ratio', type=float, default=0.5) # label missing ratio, 0.5 means 50% of the labels are missing
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--curve-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='10B_final_')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--pre_epochs', type=int, default=10)
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--sigma', type=float, default=1e-1)


    args = parser.parse_args()

    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if not os.path.exists(args.records_dir):
        os.makedirs(args.records_dir)
    datasets = args.datasets or ([args.dataset] if args.dataset else None)
    if not datasets:
        parser.error("one of --dataset or --datasets is required")
    for dataset in datasets:
        args.dataset = dataset
        file_path = args.file_path or osp.join(
            args.records_dir,
            args.name + args.dataset + '_VM_' + str(args.mask_view_ratio) +
            '_LM_' + str(args.mask_label_ratio) + '_T_' +
            str(args.training_sample_ratio) + '.txt',
        )
        main(args, file_path)
