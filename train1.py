import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import models
from utils.logger import Logger
import myexman
from utils import utils
import sys


def add_learner_params(parser):
    parser.add_argument('--problem', default='sim-clr',
                        help='The problem to train',
                        choices=models.REGISTERED_MODELS,
                        )
    parser.add_argument('--name', default='',
                        help='Name for the experiment',
                        )
    parser.add_argument('--ckpt', default='',
                        help='Optional checkpoint to init the model.'
                        )
    parser.add_argument('--verbose', default=False, type=bool)
    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-anneal')
    parser.add_argument('--opt', default='lars', help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=-1, type=int, help='The number of optimizer updates')
    parser.add_argument('--warmup', default=0, type=float, help='The number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=0.1, type=float, help='Base learning rate')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, dest='weight_decay')
    # trainer params
    parser.add_argument('--save_freq', default=10000000000000000, type=int, help='Frequency to save the model')
    parser.add_argument('--log_freq', default=100, type=int, help='Logging frequency')
    parser.add_argument('--eval_freq', default=10000000000000000, type=int, help='Evaluation frequency')
    parser.add_argument('-j', '--workers', default=4, type=int, help='The number of data loader workers')
    parser.add_argument('--eval_only', default=False, type=bool, help='Skips the training step if True')
    parser.add_argument('--seed', default=-1, type=int, help='Random seed')


def main():
    parser = myexman.ExParser(file=os.path.basename(__file__))
    add_learner_params(parser)

    is_help = False
    if '--help' in sys.argv or '-h' in sys.argv:
        sys.argv.pop(sys.argv.index('--help' if '--help' in sys.argv else '-h'))
        is_help = True

    args, _ = parser.parse_known_args(log_params=False)

    models.REGISTERED_MODELS[args.problem].add_model_hparams(parser)

    if is_help:
        sys.argv.append('--help')

    args = parser.parse_args(namespace=args)

    if args.data == 'imagenet' and args.aug == False:
        raise Exception('ImageNet models should be eval with aug=True!')

    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.gpu = 0
    device = torch.device('cuda:%d' % args.gpu)

    # create model
    model = models.REGISTERED_MODELS[args.problem](args, device=device)

    if args.ckpt != '':
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

    # Data loading code
    model.prepare_data()
    train_loader, val_loader = model.dataloaders(iters=args.iters)

    # define optimizer
    cur_iter = 0
    optimizer, scheduler = models.ssl.configure_optimizers(args, model, cur_iter - 1)

    # optionally resume from a checkpoint
    if args.ckpt and not args.eval_only:
        optimizer.load_state_dict(ckpt['opt_state_dict'])

    cudnn.benchmark = True

    continue_training = args.iters != 0
    data_time, it_time = 0, 0

    fmt = {
        'train_time': '.3f',
        'val_time': '.3f',
        'lr': '.1e',
    }
    logger = Logger('logs', base=args.root, fmt=fmt)

    while continue_training:
        train_logs = []
        model.train()
        torch.cuda.empty_cache()  # 释放未使用的缓存  
        start_time = time.time()
        for _, batch in enumerate(train_loader):
            cur_iter += 1

            batch = [x.to(device) for x in batch]
            data_time += time.time() - start_time

            logs = {}
            if not args.eval_only:
                # forward pass and compute loss
                logs = model.train_step(batch, cur_iter)
                loss = logs['loss']

                # gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            torch.cuda.empty_cache()  # 释放未使用的缓存   
            # save logs for the batch
            train_logs.append({k: utils.tonp(v) for k, v in logs.items()})

            if cur_iter % args.save_freq == 0:
                save_checkpoint(args.root, model, optimizer, cur_iter)

            if cur_iter % args.eval_freq == 0 or cur_iter >= args.iters:
                # TODO: aggregate metrics over all processes
                test_logs = []
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = [x.to(device) for x in batch]
                        # forward pass
                        logs = model.test_step(batch)
                        # save logs for the batch
                        test_logs.append(logs)
                model.train()

                test_logs = utils.agg_all_metrics(test_logs)
                logger.add_logs(cur_iter, test_logs, pref='test_')

            it_time += time.time() - start_time

            if (cur_iter % args.log_freq == 0 or cur_iter >= args.iters):
                save_checkpoint(args.root, model, optimizer)
                train_logs = utils.agg_all_metrics(train_logs)

                logger.add_logs(cur_iter, train_logs, pref='train_')
                logger.add_scalar(cur_iter, 'lr', optimizer.param_groups[0]['lr'])
                logger.add_scalar(cur_iter, 'data_time', data_time)
                logger.add_scalar(cur_iter, 'it_time', it_time)
                logger.iter_info()
                logger.save()

                data_time, it_time = 0, 0
                train_logs = []

            if scheduler is not None:
                scheduler.step()

            if cur_iter >= args.iters:
                continue_training = False
                break

            start_time = time.time()

    save_checkpoint(args.root, model, optimizer)


def save_checkpoint(path, model, optimizer, cur_iter=None):
    if cur_iter is None:
        fname = os.path.join(path, 'checkpoint.pth.tar')
    else:
        fname = os.path.join(path, 'checkpoint-%d.pth.tar' % cur_iter)

    ckpt = model.get_ckpt()
    ckpt.update(
        {
            'opt_state_dict': optimizer.state_dict(),
            'iter': cur_iter,
        }
    )

    torch.save(ckpt, fname)


if __name__ == '__main__':
    main()