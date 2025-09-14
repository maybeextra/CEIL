import torch
from code.train.trainer.base_trainer import Base_trainer


class Trainer_stage1(Base_trainer):
    def __init__(self,model, args, kind, writer = None, optimizer = None, scheduler = None, scaler = None):
        super().__init__(model, optimizer, scheduler, scaler, writer, args, kind)
        self.train_metric_names = ['Loss_All', 'Loss_RGB', 'Loss_IR', 'Loss_MMD']

    @staticmethod
    def _parser_data(item):
        if len(item) == 5:
            img = torch.cat((item[0], item[1]), 0)
            proxy = torch.cat((item[2], item[2]), 0)
            label = torch.cat((item[3], item[3]), 0)
            cam = torch.cat((item[4], item[4]), 0)
        else:
            img, proxy, label, cam = item

        img, proxy, label = [x.cuda() for x in [img, proxy, label]]

        return img, proxy, label, cam

    def run(self, rgb_trainloader, ir_trainloader, magnification = 1):
        self._pre_for_train(rgb_trainloader, ir_trainloader)
        metrics = self._init_metric(self.train_metric_names)
        for iter_index in range(self.train_iters):
            self.optimizer.zero_grad()

            img_rgb, proxy_rgb, label_rgb, cam_rgb = self._parser_data(rgb_trainloader.next())
            img_ir, proxy_ir, label_ir, cam_ir = self._parser_data(ir_trainloader.next())

            batch_size = self.batch_size * magnification
            with self.amp_autocast():
                feats = self.model(x1=img_rgb, x2=img_ir, modal=0)
                feats_rgb, feats_ir = feats[:batch_size], feats[batch_size:]

                loss_mmd = self.mmd_loss(feats_rgb, feats_ir)
                loss_proxy_rgb, loss_proxy_ir, _, _ = self.memory(
                    feats_rgb, proxy_rgb, label_rgb, None, None, None,
                    feats_ir, proxy_ir, label_ir, None, None, None
                )

                loss = loss_proxy_rgb + loss_proxy_ir + loss_mmd

            self._optimize(loss)
            self._update_train(
                i=iter_index,
                metrics = metrics,magnification = magnification,
                Loss_All=loss, Loss_RGB=loss_proxy_rgb, Loss_IR=loss_proxy_ir, Loss_MMD=loss_mmd
            )
        self.scheduler.step()
        self.pbar.close()
        self._record_metrics(metrics, 'train')