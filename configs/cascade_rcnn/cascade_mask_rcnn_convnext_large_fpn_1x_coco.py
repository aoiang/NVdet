_base_ = [
    '../_base_/models/cascade_mask_rcnn_convnext_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
    model_name='convnext_large'
        ),
    neck=dict(
    in_channels=[192, 384, 768, 1536],
    ))