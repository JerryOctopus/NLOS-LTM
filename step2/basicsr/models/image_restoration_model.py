# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import os
from os import path as osp
from tqdm import tqdm

from models.archs import define_network
from models.base_model import BaseModel
from utils import get_root_logger, imwrite, tensor2img, save_image
from utils.dist_util import get_dist_info
from models.archs.encoder import ConvEncoderLoss
from models.archs.discriminator import MultiscaleDiscriminator
from models.archs.loss import GANLoss, VGGLoss, OTLoss
import pdb

loss_module = importlib.import_module('models.archs.loss')
metric_module = importlib.import_module('metrics')

class ImageRestorationModel(BaseModel): 
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g1 = define_network(deepcopy(opt['network_g1']))
        self.net_g1 = self.model_to_device(self.net_g1)

        self.net_g2 = define_network(deepcopy(opt['network_g2']))
        self.net_g2 = self.model_to_device(self.net_g2)
        self.print_network(self.net_g2)
        
        self.net_d = MultiscaleDiscriminator()
        self.net_d = self.model_to_device(self.net_d)
        
        self.criterionGAN = GANLoss('hinge')
        self.criterionVGG = VGGLoss(0)
        self.criterionFeat = torch.nn.L1Loss()
        self.L1 = torch.nn.L1Loss()
        self.OTloss = OTLoss()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g1', None)
        if load_path is not None:
            self.load_network(self.net_g1, load_path,
                              self.opt['path'].get('strict_load_g1', True), param_key=self.opt['path'].get('param_key', 'params'))

        load_path = self.opt['path'].get('pretrain_network_g2', None)
        if load_path is not None:
            self.load_network(self.net_g2, load_path,
                              self.opt['path'].get('strict_load_g2', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
       

    def model43_to_device(self, net, find_unused_parameters):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            net (nn.Module)
        """

        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def init_training_settings(self):
        self.net_g2.train()
        train_opt = self.opt['train']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []
        for k, v in self.net_g2.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        d_optim_params = []
        d_optim_params_lowlr = []
        for k, v in self.net_d.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    d_optim_params_lowlr.append(v)
                else:
                    d_optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        self.optimizer_d = torch.optim.AdamW([{'params': d_optim_params}, {'params': d_optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_d)
        
        
        # print(self.optimizer_g)
        # exit(0)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.label = data['label'].to(self.device)
     
        

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t
    
    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq

    def compute_generator_loss(self):
        state = 'train'
        G_losses = OrderedDict() #{}
        self.net_g1.eval()
        self.net_g2.module.generator.fc3.eval()
        self.net_g2.module.generator.decoders.eval()
        self.net_g2.module.generator.ending.eval()
        gt_pred, latent_1 = self.net_g1(self.gt)
        latent_1 = latent_1.detach()
        gt_pred = gt_pred.detach()
        pred, fake_image, latent_2, vq_loss = self.net_g2(self.lq, self.gt, self.label, state)
        pred_fake, pred_real = self.discriminate(
            self.gt, fake_image, self.lq)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False) + vq_loss
        num_D = len(pred_fake)
        GAN_Feat_loss = torch.FloatTensor(1).cuda().fill_(0)
        for i in range(num_D):  # for each discriminator
            # last output is the final prediction, so we exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach()) 
                GAN_Feat_loss += unweighted_loss * 10 / num_D
        G_losses['GAN_Feat'] = GAN_Feat_loss
        G_losses['VGG'] = self.criterionVGG(fake_image, self.lq)*30
        
        l_pix = 0.
        l_pix += 10 *self.L1(pred, self.gt)

            # print('l pix ... ', l_pix)
        G_losses['l_pix'] = l_pix
        
        G_losses['OT'] = self.OTloss(latent_2, latent_1) * 0.009
        return gt_pred, pred, fake_image, G_losses
                                                                                             
        
        
    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1) 

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.net_d(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
    def compute_discriminator_loss(self):
        state = 'train'
        D_losses = OrderedDict() # {}
        with torch.no_grad():
            _, fake_image, _, _ = self.net_g2(self.lq, self.gt, self.label, state)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(
            self.gt, fake_image, self.lq)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True) 
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True) 
        return D_losses


    def optimize_parameters(self, current_iter):
        self.net_g1.eval()
        self.optimizer_g.zero_grad()
        gt_preds, preds, inverse_preds, G_losses = self.compute_generator_loss()
        #if not isinstance(preds, list):
        #     preds = [preds]
        self.step1 = gt_preds
        self.output = preds
        self.output1 = inverse_preds
        l_total = (G_losses['GAN'] + G_losses['GAN_Feat'] + G_losses['VGG']) + G_losses['l_pix'] + G_losses['OT']
        l_total = l_total + 0 * sum(p.sum() for p in self.net_g2.parameters())
        l_total.backward()
        # use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        # if use_grad_clip:
        #     print("use grad clip", use_grad_clip)
        torch.nn.utils.clip_grad_norm_(self.net_g2.parameters(), 0.01)
        # exit(0)
        self.optimizer_g.step()
        
        self.optimizer_d.zero_grad()
        D_losses = self.compute_discriminator_loss()
        D_loss = sum(D_losses.values()).mean()
        D_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 0.01)
        self.optimizer_d.step()
        # exit(0)
        # whole_loss = {}
        # whole_loss.update(G_losses)
        # whole_loss.update(D_losses)
        # print(G_losses)
        
        self.log_g_dict = G_losses
        # exit(0)
        self.log_d_dict = D_losses

    def val(self):
        state = 'val'
        self.net_g1.eval()
        self.net_g2.eval()
        with torch.no_grad():
            n = self.lq.size(0) # batch_size
            st1 = []
            outs = []
            outs1 = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                gt_pred, latent_1 = self.net_g1(self.gt[i:j, :, :, :])
                latent_1 = latent_1.detach()
                gt_pred = gt_pred.detach()

                pred, inverse_pred, _, vq_loss = self.net_g2(self.lq[i:j, :, :, :], self.gt[i:j, :, :, :], self.label, state)
                if isinstance(gt_pred, list):
                    gt_pred = gt_pred[-1]
                st1.append(gt_pred)

                if isinstance(pred, list):
                    pred = pred[-1]
                # print('pred .. size', pred.size())
                outs.append(pred)

                if isinstance(inverse_pred, list):
                    inverse_pred = inverse_pred[-1]
                outs1.append(inverse_pred)
                i = j
            self.mistake = vq_loss

            self.step1 = torch.cat(st1, dim=0)
            self.output = torch.cat(outs, dim=0)
            self.output1 = torch.cat(outs1, dim=0)
        self.net_g2.train()
        self.net_g2.module.generator.fc3.eval()
        self.net_g2.module.generator.decoders.eval()
        self.net_g2.module.generator.ending.eval()

    def test(self):
        state = 'test'
        self.net_g2.eval()
        with torch.no_grad():
            n = self.lq.size(0) # batch_size
            outs = []
            outs1 = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n: # i=0,n=1
                j = i + m
                if j >= n:
                    j = n

                pred, inverse_pred, _, vq_loss = self.net_g2(self.lq[i:j, :, :, :], self.gt[i:j, :, :, :], self.label[i:j], state)
                if isinstance(pred, list):
                    pred = pred[-1]
                # print('pred .. size', pred.size())
                outs.append(pred)

                if isinstance(inverse_pred, list):
                    inverse_pred = inverse_pred[-1]
                outs1.append(inverse_pred)
                i = j
            self.mistake = vq_loss

            self.step1 = None
            self.output = torch.cat(outs, dim=0)
            self.output1 = torch.cat(outs1, dim=0)
        self.net_g2.train()
        self.net_g2.module.generator.fc3.eval()
        self.net_g2.module.generator.decoders.eval()
        self.net_g2.module.generator.ending.eval()


    def get_latest_images(self):
        if hasattr(self, 'gt'):
            return [self.lq, self.step1, self.output, self.output1, self.gt]
        else:
            return [self.lq, self.step1, self.output, self.output1]



    def dist_validation(self, dataloader, visualizer, epoch, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, visualizer, epoch, current_iter, epoch_steps, len_trainset, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, visualizer, epoch, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results_s1 = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_reblur = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        
        self.mistake = 0

        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        count = 0

        for idx, val_data in enumerate(dataloader):
            count=count+1
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # if img_name[-1] != '9':
            #     continue

            # print('val_data .. ', val_data['lq'].size(), val_data['gt'].size())
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.val()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_latest_images()
            results = OrderedDict()
            lq_img = tensor2img(visuals[0].data)
            s1_img = tensor2img(visuals[1].data)
            sr_img = tensor2img(visuals[2].data)
            re_img = tensor2img(visuals[3].data) 
            if len(visuals)==4:
                results = OrderedDict([('Input', lq_img), ('Gtres', s1_img), ('Deblur', sr_img), ('Reblur', re_img)])  
            else:
                gt_img =  tensor2img(visuals[4].data) # gt   
                results = OrderedDict([('Input', lq_img), ('Gtres', s1_img), ('Deblur', sr_img), ('Reblur', re_img), ('Gt', gt_img)])      
                del self.gt     

            # tentative for out of GPU memory
            del self.lq
            del self.step1
            del self.output
            del self.output1
            torch.cuda.empty_cache()

            if(count % self.opt['val']['display_freq']==0):
                visualizer.display_current_results(results, epoch)
            if(count==500):
                break


            if save_img:
                
                if self.opt['is_train']:
                    save_lq_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_lq.png')

                    save_s1_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_s1.png')
                    
                   
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_de.png')
                    save_re_img_path = osp.join(self.opt['path']['visualization'],
                            img_name,
                            f'{img_name}_{current_iter}_re.png')                           
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    save_lq_img_path = osp.join(
                        self.opt['path']['visualization'], 
                        f'{img_name}_lq.png')
                    save_s1_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_s1.png')
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'],
                        f'{img_name}.png')
                    save_re_img_path = osp.join(
                        self.opt['path']['visualization'], 
                        f'{img_name}_re.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], 
                        f'{img_name}_gt.png')

                save_image(lq_img, save_lq_img_path)
                save_image(s1_img, save_s1_img_path)              
                save_image(sr_img, save_img_path)
                save_image(re_img, save_re_img_path)  
                save_image(gt_img, save_gt_img_path)
                    

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results_s1[name] += getattr(
                        metric_module, metric_type)(s1_img, gt_img, **opt_)
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(sr_img, gt_img, **opt_)
                    self.metric_results_reblur[name] += getattr(
                        metric_module, metric_type)(re_img, lq_img, **opt_)
                        

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1

        pbar.close()

        current_metric_s1 = 0.
        current_metric = 0.
        current_metric_reblur = 0.
        if with_metrics:
            for metric in self.metric_results_s1.keys():
                self.metric_results_s1[metric] /= cnt
                current_metric_s1 = self.metric_results_s1[metric]
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            for metric in self.metric_results_reblur.keys():
                self.metric_results_reblur[metric] /= cnt
                current_metric_reblur = self.metric_results_reblur[metric]

            self.mistake /= cnt



            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric_s1, current_metric, current_metric_reblur, self.mistake



    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        #pdb.set_trace()
        log_str_s1 = f'Validation_s1 {dataset_name},\t'
        log_str = f'Validation {dataset_name},\t'
        log_str_reblur = f'Validation_reblur {dataset_name},\t'


        for metric, value in self.metric_results_s1.items():
            log_str_s1 += f'\t # {metric}: {value:.4f}'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        for metric, value in self.metric_results_reblur.items():
            log_str_reblur += f'\t # {metric}: {value:.4f}'



        logger_s1 = get_root_logger()
        logger_s1.info(log_str_s1)

        logger = get_root_logger()
        logger.info(log_str)

        logger_reblur = get_root_logger()
        logger_reblur.info(log_str_reblur)


        if tb_logger:
            for metric, value in self.metric_results_s1.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
            for metric, value in self.metric_results_reblur.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)



    def _log_test_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        #pdb.set_trace()
        log_str = f'Test {dataset_name},\t'
        log_str_reblur = f'Test_reblur {dataset_name},\t'

        log_str_mistake = f'Test_vq {dataset_name},\t'

        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        for metric, value in self.metric_results_reblur.items():
            log_str_reblur += f'\t # {metric}: {value:.4f}'

        log_str_mistake += f'\t # mistake: {self.mistake:.4f}'

        logger = get_root_logger()
        logger.info(log_str)

        logger_reblur = get_root_logger()
        logger_reblur.info(log_str_reblur)

        logger_mistake = get_root_logger()
        logger_mistake.info(log_str_mistake)

        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
            for metric, value in self.metric_results_reblur.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)



    def validation_test(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            self.metric_results_reblur = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # if img_name[-1] != '9':
            #     continue

            # print('val_data .. ', val_data['lq'].size(), val_data['gt'].size())
            self.feed_data(val_data)
            if self.opt['val'].get('grids', False):
                self.grids()

            #self.test()
            self.val()
            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_latest_images()
            results = OrderedDict()
            lq_img = tensor2img(visuals[0].data)
            s1_img = tensor2img(visuals[1].data)
            sr_img = tensor2img(visuals[2].data)
            re_img = tensor2img(visuals[3].data) 


            if len(visuals)==4:
                results = OrderedDict([('Input', lq_img), ('Gtres', s1_img), ('Deblur', sr_img), ('Reblur', re_img)])  
                #results = OrderedDict([('Input', lq_img), ('Deblur', sr_img), ('Reblur', re_img)])  
            else:
                gt_img =  tensor2img(visuals[4].data) # gt   
                #results = OrderedDict([('Input', lq_img), ('Deblur', sr_img), ('Reblur', re_img), ('Gt', gt_img)])      
                results = OrderedDict([('Input', lq_img), ('Gtres', s1_img), ('Deblur', sr_img), ('Reblur', re_img), ('Gt', gt_img)])                      
                del self.gt     

            # tentative for out of GPU memory
            del self.lq
            del self.step1
            del self.output
            del self.output1
            torch.cuda.empty_cache()


            if save_img:
                
                if self.opt['is_train']:                    
                    save_lq_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_lq.png')
                    save_s1_img_path = osp.join(self.opt['path']['visualization'],
                                        img_name,
                                        f'{img_name}_{current_iter}_s1.png')
               
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_de.png')
                    save_re_img_path = osp.join(self.opt['path']['visualization'],
                            img_name,
                            f'{img_name}_{current_iter}_re.png')                           
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:                    
                    save_lq_img_path = osp.join(
                        self.opt['path']['visualization'], 
                        f'{img_name}_lq.png')
                    save_s1_img_path = osp.join(
                        self.opt['path']['visualization'], f'{img_name}_s1.png')

                    save_img_path = osp.join(
                        self.opt['path']['visualization'], f'{img_name}.png')
                    save_re_img_path = osp.join(
                        self.opt['path']['visualization'], f'{img_name}_re.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], f'{img_name}_gt.png')


                save_image(lq_img, save_lq_img_path)
                save_image(s1_img, save_s1_img_path)              
                save_image(sr_img, save_img_path)
                save_image(re_img, save_re_img_path)  
                save_image(gt_img, save_gt_img_path)
           
            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(sr_img, gt_img, **opt_)
                    self.metric_results_reblur[name] += getattr(
                        metric_module, metric_type)(re_img, lq_img, **opt_)


            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
            # if cnt == 300:
            #     break
        pbar.close()

        current_metric = 0.
        current_metric_reblur = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]
            for metric in self.metric_results_reblur.keys():
                self.metric_results_reblur[metric] /= cnt
                current_metric_reblur = self.metric_results_reblur[metric]


            self._log_test_metric_values(current_iter, dataset_name,
                                               tb_logger)
            self.mistake /= cnt

        return current_metric, current_metric_reblur, self.mistake


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['Gtres'] = self.step1.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['inverse'] = self.output1.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g2, 'net_g2', current_iter)
        self.save_training_state(epoch, current_iter)