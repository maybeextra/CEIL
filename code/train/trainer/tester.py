import copy
import logging

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from code.tools import rerank
from code.tools.eavl.get_test_data import creat_test_data
from code.train.trainer.base_trainer import Base_trainer
from code.tools import eavl as e
import time

class Tester(Base_trainer):
    def __init__(self, model, args, kind=None, writer=None, optimizer=None, scheduler=None, scaler=None):
        super().__init__(model, optimizer, scheduler, scaler, writer, args, kind)
        self.rerank = rerank.creat(args.reranking_type)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def denormalize(self, image):
        image_c = copy.deepcopy(image)
        for c, t in enumerate(image_c):
            image_c[c] = t * self.std[c] + self.mean[c]
        return image_c.transpose(1, 2, 0)

    def visualize_retrieved_images(self, query_img, retrieved_imgs, matches, save_path='retrieved_images.png'):
        fig, axes = plt.subplots(1, len(retrieved_imgs) + 1, figsize=(15, 5))

        # Adjust space between images
        fig.subplots_adjust(wspace=0.3)

        query_img = self.denormalize(query_img)
        axes[0].imshow(query_img)
        axes[0].axis('off')

        for i, (img, match) in enumerate(zip(retrieved_imgs, matches)):
            img = self.denormalize(img)
            axes[i + 1].imshow(img)
            axes[i + 1].axis('off')

            color = 'green' if match else 'red'
            rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0], linewidth=5, edgecolor=color, facecolor='none')
            axes[i + 1].add_patch(rect)

        # Save the figure without borders
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def draw(self, args, test_mode, mode):
        logging.info(f'Test mode: {test_mode} | mode: {mode}')

        query_loader = creat_test_data(args, mode=mode, kind='query')
        query_feat, query_label, query_cam, query_img = self.extract_features_test(query_loader, test_mode[0], False, True)

        gall_loader = creat_test_data(args, 1, mode=mode, kind='select')
        gall_feat, gall_label, gall_cam, gall_img = self.extract_features_test(gall_loader, test_mode[1], False, True)
        print(len(query_feat), len(gall_feat))

        num = 2
        query_img = query_img[num]
        q_id = query_label[num]
        print(q_id)

        dist = -torch.matmul(query_feat, gall_feat.T).cpu().numpy()
        top_retrieved_idxs = np.argsort(dist[num])[:10]
        g_id = gall_label[top_retrieved_idxs]
        print(g_id)

        matches = (g_id == q_id).astype(np.int32)
        top_retrieved_imgs = [gall_img[top_retrieved_idxs[i]] for i in range(10)]
        self.visualize_retrieved_images(query_img, top_retrieved_imgs, matches, f'retrieved_CERL_{num}.png')

        dist = -self.rerank(query_feat, gall_feat, query_cam, gall_cam, k1=args.gnn_k1, k2=args.gnn_k2)
        top_retrieved_idxs = np.argsort(dist[num])[:10]
        g_id = gall_label[top_retrieved_idxs]
        print(g_id)

        matches = (g_id == q_id).astype(np.int32)
        top_retrieved_imgs = [gall_img[top_retrieved_idxs[i]] for i in range(10)]
        self.visualize_retrieved_images(query_img, top_retrieved_imgs, matches, f'retrieved_reranking_{num}.png')

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
            query_loader = creat_test_data(args, mode=mode, kind='query')
            query_feat, query_label, query_cam = self.extract_features_test(query_loader, test_mode[0], boost=args.test_boost)

        all_metrics = [None] * 6  # all_cmc_1, all_cmc_2, all_mAP_1, all_mAP_2, all_mINP_1, all_mINP_2

        all_time = 0
        for trial in range(1, num_iter + 1):
            gall_loader = creat_test_data(args, trial, mode=mode, kind='gallery')
            gall_feat, gall_label, gall_cam = self.extract_features_test(gall_loader, test_mode[1], boost=args.test_boost)

            if args.dataset == 'RegDB':
                query_loader = creat_test_data(args, trial=trial, mode=mode, kind='query')
                query_feat, query_label, query_cam = self.extract_features_test(query_loader, test_mode[0], boost=args.test_boost)

            dist = -torch.matmul(query_feat, gall_feat.T).cpu().numpy()
            cmc_1, mAP_1, mINP_1 = e.use(args.dataset, dist, query_label, gall_label, query_cam, gall_cam)

            start_time = time.time()
            dist = -self.rerank(query_feat, gall_feat, query_cam, gall_cam, k1=args.gnn_k1, k2=args.gnn_k2)
            elapsed_time = time.time() - start_time
            all_time += elapsed_time
            cmc_2, mAP_2, mINP_2 = e.use(args.dataset, dist, query_label, gall_label, query_cam, gall_cam)

            logging.info(f'Test Trial: {trial}, Elapsed time: {elapsed_time:.3}s')
            logging.info(f"Performance: Rank-1: {cmc_1[0]:.2%} | Rank-5: {cmc_1[4]:.2%} | Rank-10: {cmc_1[9]:.2%}| Rank-20: {cmc_1[19]:.2%}| mAP: {mAP_1:.2%}| mINP: {mINP_1:.2%}")
            logging.info(f"R Performance: Rank-1: {cmc_2[0]:.2%} | Rank-5: {cmc_2[4]:.2%} | Rank-10: {cmc_2[9]:.2%}| Rank-20: {cmc_2[19]:.2%}| mAP: {mAP_2:.2%}| mINP: {mINP_2:.2%}")
            logging.info("-----------------------Next Trial--------------------")

            current_metrics = [cmc_1, cmc_2, mAP_1, mAP_2, mINP_1, mINP_2]
            all_metrics = self._update_test(trial, all_metrics, current_metrics)


        all_metrics = [metric / num_iter for metric in all_metrics]

        logging.info("---------------All Performance---------------")
        logging.info(f'Re Ranking time: {float(all_time)*1000:.4f}ms')
        logging.info(f'All Average:')
        logging.info(f'Performance: Rank-1: {all_metrics[0][0]:.2%} | Rank-5: {all_metrics[0][4]:.2%} | Rank-10: {all_metrics[0][9]:.2%}| Rank-20: {all_metrics[0][19]:.2%}| mAP: {all_metrics[2]:.2%}| mINP: {all_metrics[4]:.2%}')
        logging.info(f'R Performance: Rank-1: {all_metrics[1][0]:.2%} | Rank-5: {all_metrics[1][4]:.2%} | Rank-10: {all_metrics[1][9]:.2%}| Rank-20: {all_metrics[1][19]:.2%}| mAP: {all_metrics[3]:.2%}| mINP: {all_metrics[5]:.2%}')
        logging.info('End Test')
        logging.info('---------------------------------------------')
