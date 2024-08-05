from code.tools.eavl.eval_metrics import eval_sysu, eval_regdb, eval_llcm
from code.tools.eavl.get_test_data import creat_test_data_query,creat_test_data_gall
import logging
import torch
from code.train.trainer.base_trainer import Base_trainer
from code.tools import rerank

class Tester(Base_trainer):
    def __init__(self, model, args, kind=None, writer=None, optimizer=None, scheduler=None, scaler=None):
        super().__init__(model, optimizer, scheduler, scaler, writer, args, kind)
        self.rerank = rerank.creat(args.reranking_type)

    @staticmethod
    def _update_test(trial, all_metrics, current_metrics):
        if trial == 1:
            return current_metrics
        else:
            return [all_val + curr_val for all_val, curr_val in zip(all_metrics, current_metrics)]

    def test(self, args, test_mode, mode):
        num_iter = int(args.eval_iter)
        logging.info(f'Test mode: {test_mode} | mode: {mode}')

        if args.dataset != 'RegDB':
            query_loader = creat_test_data_query(args, mode=mode)
            query_feat, query_label, query_cam = self.extract_features(query_loader, test_mode[0], False)

        all_metrics = [None] * 6  # all_cmc_1, all_cmc_2, all_mAP_1, all_mAP_2, all_mINP_1, all_mINP_2

        for trial in range(1, num_iter + 1):
            gall_loader = creat_test_data_gall(args, trial, mode=mode)
            gall_feat, gall_label, gall_cam = self.extract_features(gall_loader, test_mode[1], False)

            if args.dataset == 'SYSU':
                _eval = eval_sysu
            elif args.dataset == 'RegDB':
                query_loader = creat_test_data_query(args, trial, mode=mode)
                query_feat, query_label, query_cam = self.extract_features(query_loader, test_mode[0], False)
                _eval = eval_regdb
            elif args.dataset == 'LLCM':
                _eval = eval_llcm
            else:
                raise RuntimeError('Unknown dataset')

            dist = -torch.matmul(query_feat, gall_feat.T).cpu().numpy()
            cmc_1, mAP_1, mINP_1 = _eval(dist, query_label, gall_label, query_cam, gall_cam)

            dist = -self.rerank(query_feat, gall_feat, query_cam, gall_cam, k1=args.gnn_k1, k2=args.gnn_k2)
            cmc_2, mAP_2, mINP_2 = _eval(dist, query_label, gall_label, query_cam, gall_cam)

            logging.info(f'Test Trial: {trial}')
            logging.info(f"Performance: Rank-1: {cmc_1[0]:.2%} | Rank-5: {cmc_1[4]:.2%} | Rank-10: {cmc_1[9]:.2%}| Rank-20: {cmc_1[19]:.2%}| mAP: {mAP_1:.2%}| mINP: {mINP_1:.2%}")
            logging.info(f"R Performance: Rank-1: {cmc_2[0]:.2%} | Rank-5: {cmc_2[4]:.2%} | Rank-10: {cmc_2[9]:.2%}| Rank-20: {cmc_2[19]:.2%}| mAP: {mAP_2:.2%}| mINP: {mINP_2:.2%}")
            logging.info("-----------------------Next Trial--------------------")

            current_metrics = [cmc_1, cmc_2, mAP_1, mAP_2, mINP_1, mINP_2]
            all_metrics = self._update_test(trial, all_metrics, current_metrics)

        all_metrics = [metric / num_iter for metric in all_metrics]

        logging.info("---------------All Performance---------------")
        logging.info('All Average:')
        logging.info(f'Performance: Rank-1: {all_metrics[0][0]:.2%} | Rank-5: {all_metrics[0][4]:.2%} | Rank-10: {all_metrics[0][9]:.2%}| Rank-20: {all_metrics[0][19]:.2%}| mAP: {all_metrics[2]:.2%}| mINP: {all_metrics[4]:.2%}')
        logging.info(f'R Performance: Rank-1: {all_metrics[1][0]:.2%} | Rank-5: {all_metrics[1][4]:.2%} | Rank-10: {all_metrics[1][9]:.2%}| Rank-20: {all_metrics[1][19]:.2%}| mAP: {all_metrics[3]:.2%}| mINP: {all_metrics[5]:.2%}')
        logging.info('End Test')
        logging.info('---------------------------------------------')
