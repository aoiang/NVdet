# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_root_logger, setup_multi_processes,
                         update_data_root)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def export_onnxfile(cfg, model):
    # print(model)
    import torch.onnx

    # datasets = [build_dataset(cfg.data.train)]
    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     val_dataset.pipeline = cfg.data.train.pipeline
    #     datasets.append(build_dataset(val_dataset))
    # if cfg.checkpoint_config is not None:
    #     # save mmdet version, config file content and class names in
    #     # checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         mmdet_version=__version__ + get_git_hash()[:7],
    #         CLASSES=datasets[0].CLASSES)
    # # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES
    # model.num_classes = len(datasets[0].CLASSES)

    from torch.utils.tensorboard import SummaryWriter
    from mmcv.cnn import get_model_complexity_info

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')


    backbone = model.backbone
    neck = model.neck
    head = model.bbox_head



    '''
    yolox input size

    backbone: (8, 3, 640, 640)
    head: ([8, 96, 80, 80], [8, 192, 40, 40], [8, 384, 20, 20])
    neck: ([8, 96, 80, 80], [8, 96, 40, 40], [8, 96, 20, 20])


    '''
    # export_onnx = 'backbone'
    export_onnx = 'neck'
    # export_onnx = 'head'
    # export_onnx = 'entire'
    name = 'yolox'
    s = 'tiny'
    # s = 'small'
    # s = 'large'
    # s = 'xlarge'
    y = torch.randn(1, 3, 640, 640, requires_grad=True)

    from pthflops import count_ops

    if export_onnx == 'backbone':
        x = torch.randn(1, 3, 640, 640, requires_grad=True)
        m = backbone
    elif export_onnx == 'entire':
        x = torch.randn(1, 3, 640, 640, requires_grad=True)
        m = model
    elif export_onnx == 'neck':
        outs = []
        # x1 = torch.randn(1, 96, 100, 100, requires_grad=True)
        # x2 = torch.randn(1, 192, 50, 50, requires_grad=True)
        # x3 = torch.randn(1, 384, 25, 25, requires_grad=True)

        # x1 = torch.randn(1, 128, 100, 100, requires_grad=True)
        # x2 = torch.randn(1, 256, 50, 50, requires_grad=True)
        # x3 = torch.randn(1, 512, 25, 25, requires_grad=True)

        # x1 = torch.randn(1, 256, 100, 100, requires_grad=True)
        # x2 = torch.randn(1, 512, 50, 50, requires_grad=True)
        # x3 = torch.randn(1, 1024, 25, 25, requires_grad=True)
        #
        # x1 = torch.randn(1, 320, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 640, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 1280, 20, 20, requires_grad=True)

        x1 = torch.randn(1, 96, 80, 80, requires_grad=True)
        x2 = torch.randn(1, 192, 40, 40, requires_grad=True)
        x3 = torch.randn(1, 384, 20, 20, requires_grad=True)

        # x1 = torch.randn(1, 128, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 256, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 512, 20, 20, requires_grad=True)
        #
        # x1 = torch.randn(1, 256, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 512, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 1024, 20, 20, requires_grad=True)
        #
        # x1 = torch.randn(1, 320, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 640, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 1280, 20, 20, requires_grad=True)







        outs.append(x1)
        outs.append(x2)
        outs.append(x3)
        x = tuple(outs)
        # x = tuple(outs)

        m = neck
    else:
        x1 = torch.randn(1, 96, 100, 100, requires_grad=True)
        x2 = torch.randn(1, 96, 50, 50, requires_grad=True)
        x3 = torch.randn(1, 96, 25, 25, requires_grad=True)

        # x1 = torch.randn(1, 128, 100, 100, requires_grad=True)
        # x2 = torch.randn(1, 128, 50, 50, requires_grad=True)
        # x3 = torch.randn(1, 128, 25, 25, requires_grad=True)

        # x1 = torch.randn(1, 256, 100, 100, requires_grad=True)
        # x2 = torch.randn(1, 256, 50, 50, requires_grad=True)
        # x3 = torch.randn(1, 256, 25, 25, requires_grad=True)
        #
        # x1 = torch.randn(1, 320, 100, 100, requires_grad=True)
        # x2 = torch.randn(1, 320, 50, 50, requires_grad=True)
        # x3 = torch.randn(1, 320, 25, 25, requires_grad=True)




        # x1 = torch.randn(1, 96, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 96, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 96, 20, 20, requires_grad=True)

        # x1 = torch.randn(1, 128, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 128, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 128, 20, 20, requires_grad=True)

        # x1 = torch.randn(1, 256, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 256, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 256, 20, 20, requires_grad=True)
        #
        # x1 = torch.randn(1, 320, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 320, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 320, 20, 20, requires_grad=True)

















        # x1 = torch.randn(1, 96, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 192, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 384, 20, 20, requires_grad=True)

        # x1 = torch.randn(1, 128, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 256, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 512, 20, 20, requires_grad=True)

        # x1 = torch.randn(1, 256, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 512, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 1024, 20, 20, requires_grad=True)

        # x1 = torch.randn(1, 320, 80, 80, requires_grad=True)
        # x2 = torch.randn(1, 640, 40, 40, requires_grad=True)
        # x3 = torch.randn(1, 1280, 20, 20, requires_grad=True)



        x = tuple([x1, x2, x3])
        m = head

    def prepare_input(resolution):
        res1 = resolution
        res2 = (resolution[0] * 2, resolution[1] // 2, resolution[2] // 2)
        res3 = (resolution[0] * 4, resolution[1] // 4, resolution[2] // 4)
        x1 = torch.FloatTensor(1, *res1)
        x2 = torch.FloatTensor(1, *res2)
        x3 = torch.FloatTensor(1, *res3)
        return dict(x1=x1, x2=x2, x3=x3)

    print(m)
    print('model parameters is', count_parameters(m))
    flops, params = get_model_complexity_info(m, (96, 80, 80), input_constructor=prepare_input)
    print(flops)
    writer.add_graph(m, x)
    writer.close()

    # print('==========here is the model==========')
    # print(m)
    # print(type(x))
    # torch.onnx.export(m,  # model being run
    #                   x,  # model input (or a tuple for multiple inputs)
    #                   # "./onnx_files/" + name + '/' + s + '/' + export_onnx + '.onnx',
    #                   "./onnx_files/relu/" + s + '/' + export_onnx + '.onnx',
    #                   # where to save the model (can be a file or file-like object)
    #                   export_params=True,  # store the trained parameter weights inside the model file
    #                   opset_version=11,  # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names=['input'],  # the model's input names
    #                   output_names=['output'],  # the model's output names
    #                   dynamic_axes={'input': {3: 'batch_size'},  # variable length axes
    #                                 'output': {3: 'batch_size'}})

    return



def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))


    # print(model.backbone)
    export_onnxfile(cfg, model)
    return


    model.init_weights()





    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    model.num_classes = len(datasets[0].CLASSES)
    print('1st size is', len(datasets[0].CLASSES))
    print('datasets 0 size is', datasets[0].CLASSES)
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
