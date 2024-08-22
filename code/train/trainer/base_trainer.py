from tqdm import tqdm
import time
from code.tools.output.log_utils import AverageMeter
from code.tools.eavl.eval_metrics import eval_sysu, eval_regdb, eval_llcm
from code.tools.eavl.get_test_data import creat_test_data
import torch
import code.models as models
import random
import copy

class Base_trainer:
    def __init__(self, model, optimizer, scheduler, scaler, writer, args, kind):
        self.epoch = None
        self.batch_time, self.start = None, None
        self.pbar = None
        self.train_iters = None

        self.model = model
        self.amp_autocast = torch.cuda.amp.autocast

        if kind == 'train':
            self.writer = writer
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.scaler = scaler
            self.batch_size = args.train_batch_size
            self.memory = models.create('memory',args)
            self.mmd_loss = MMD_loss()
            self.update_iter = args.update_iter
            self.valid_metric_name = ['cmc_1', 'cmc_5', 'cmc_10', 'cmc_20', 'mAP', 'mINP']
        elif kind == 'visualize':
            self.update_iter = args.update_iter

    def extract_features(self, data_loader, modal, flag = True, return_image = False):
        test_model = copy.deepcopy(self.model)
        test_model.eval()

        current_idx = 0
        num_samples = len(data_loader.dataset)
        features = torch.zeros((num_samples, test_model.num_features)).cuda()
        cameras = torch.zeros(num_samples, dtype=torch.int64)
        labels = torch.zeros(num_samples, dtype=torch.int64)
        images = torch.zeros((num_samples, 3, 288, 144))

        if flag:
            loop = tqdm(desc=f"Extract_features", total=len(data_loader), leave=True, ncols=160)


        with torch.no_grad():
            if flag:
                num_update_iter = len(data_loader) / self.update_iter

            for i, (img, label, cam) in enumerate(data_loader):
                batch_size = img.shape[0]  # 获取当前批次的大小
                if return_image:
                    images[current_idx:current_idx + batch_size, :] = img
                img = img.cuda()

                with self.amp_autocast():
                    feature= test_model(img, img, modal=modal)

                features[current_idx:current_idx + batch_size, :] = feature

                cameras[current_idx:current_idx + batch_size] = cam
                labels[current_idx:current_idx + batch_size] = label
                current_idx += batch_size

                if flag:
                    if (i / self.update_iter != num_update_iter) and (i % self.update_iter == 0) and i != 0:
                        loop.update(self.update_iter)
                    elif i == (len(data_loader)-1):
                        update = len(data_loader) % self.update_iter if len(data_loader) % self.update_iter != 0 else self.update_iter
                        loop.update(update)
        if flag:
            loop.close()

        if return_image:
            return features, labels.numpy(), cameras.numpy(), images.numpy()
        return features, labels.numpy(), cameras.numpy(),

    @staticmethod
    def _init_metric(metric_names):
        metric = {name: AverageMeter() for name in metric_names}
        return metric

    def _pre_for_train(self, rgb_trainloader, ir_trainloader):
        self.writer.add_scalar('train/Learn_rate', self.optimizer.param_groups[0]['lr'], self.epoch)
        self.start = time.time()
        self.batch_time = AverageMeter()

        self.model.train()

        len_of_rgb = len(rgb_trainloader)
        len_of_ir = len(ir_trainloader)
        self.train_iters = max(len_of_rgb, len_of_ir)
        self.num_update_iter = self.train_iters / self.update_iter
        self.pbar = tqdm(desc=f"Train", total=self.train_iters, leave=True, ncols=160)

    def _optimize(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _update_train(self, i, metrics, magnification = 1, **losses):
        self.batch_time.update(time.time() - self.start)
        self.start = time.time()

        loss_names = list(losses.keys())
        loss_values = list(losses.values())

        for name, value in zip(loss_names, loss_values):
            metrics[name].update(value)

        if (i / self.update_iter != self.num_update_iter) and (i % self.update_iter == 0) and i != 0:
            loss_postfix = " ".join(f"({metrics[name].avg:.3f})" for name in loss_names)

            self.pbar.set_postfix(Speed=f"{self.batch_size * magnification * 2 / self.batch_time.avg:.1f} samples/s",
                                  Loss=loss_postfix,
                                  lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
            self.pbar.update(self.update_iter)
        elif i == (self.train_iters - 1):
            loss_postfix = " ".join(f"({metrics[name].avg:.3f})" for name in loss_names)

            self.pbar.set_postfix(Speed=f"{self.batch_size * magnification * 2 / self.batch_time.avg:.1f} samples/s",
                                  Loss=loss_postfix,
                                  lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
            update = self.train_iters % self.update_iter if self.train_iters % self.update_iter != 0 else self.update_iter
            self.pbar.update(update)

    def _update_valid(self, metrics, **vals):
        val_names = list(vals.keys())
        val_values = list(vals.values())
        for name, value in zip(val_names, val_values):
            metrics[name].update(value)
        val_postfix = " ".join(f"{name}:{metrics[name].avg:.2%} " for name in val_names)
        self.pbar.set_postfix(vals=val_postfix)
        self.pbar.update()

    def _record_metrics(self, metrics, kind):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{kind}/{key}', value.avg, self.epoch)

    def valid(self, args, test_mode, mode):
        metrics = self._init_metric(self.valid_metric_name)
        num_iter = int(args.eval_iter)
        self.pbar = tqdm(desc=f"Test:", total=num_iter, leave=True, ncols=160)

        if args.dataset != 'RegDB':
            query_loader = creat_test_data(args, mode=mode, kind='query')
            query_feat, query_label, query_cam = self.extract_features(query_loader, test_mode[0], False)

        for trial in range(1, num_iter+1):
            gall_loader = creat_test_data(args, trial=trial, mode=mode, kind='gallery')
            gall_feat, gall_label, gall_cam = self.extract_features(gall_loader, test_mode[1], False)

            if args.dataset == 'SYSU':
                _eval = eval_sysu

            elif args.dataset == 'RegDB':
                query_loader = creat_test_data(args, trial=trial, mode=mode, kind='query')
                query_feat, query_label, query_cam = self.extract_features(query_loader, test_mode[0], False)
                _eval = eval_regdb

            elif args.dataset == 'LLCM':
                _eval = eval_llcm

            else:
                raise

            dist = -torch.matmul(query_feat, gall_feat.T).cpu().numpy()
            cmc, mAP, mINP = _eval(dist, query_label, gall_label, query_cam, gall_cam)

            self._update_valid(
                metrics,
                cmc_1=cmc[0], cmc_5=cmc[4], cmc_10=cmc[9], cmc_20=cmc[19],
                mAP=mAP, mINP=mINP
            )

        random.seed(args.seed)
        self.pbar.close()
        self._record_metrics(metrics, f'val/{mode}')
        return metrics['cmc_1'].avg