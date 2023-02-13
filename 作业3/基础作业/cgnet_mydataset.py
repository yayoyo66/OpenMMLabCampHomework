norm_cfg = dict(type='SyncBN', eps=0.001, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='CGNet',
        norm_cfg=dict(type='SyncBN', eps=0.001, requires_grad=True),
        in_channels=3,
        num_channels=(32, 64, 128),
        num_blocks=(3, 21),
        dilations=(2, 4),
        reductions=(8, 16)),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', eps=0.001, requires_grad=True),
        loss_decode=dict(
            type='DiceLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1, 1])),
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode='whole'))
dataset_type = 'mydataset'
data_root = '/HOME/scz0aua/run/mmsegmentation-master/dataset'
img_norm_cfg = dict(
    mean=[72.39239876, 82.90891754, 73.15835921], std=[1, 1, 1], to_rgb=True)
crop_size = (680, 680)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(680, 680), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(680, 680)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[72.39239876, 82.90891754, 73.15835921],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='Pad', size=(680, 680), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(680, 680),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[72.39239876, 82.90891754, 73.15835921],
                std=[1, 1, 1],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='mydataset',
        data_root='/HOME/scz0aua/run/mmsegmentation-master/dataset',
        img_dir=
        '/HOME/scz0aua/run/mmsegmentation-master/dataset/aug_data/aug_data/images',
        ann_dir=
        '/HOME/scz0aua/run/mmsegmentation-master/dataset/aug_data/aug_data/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(680, 680), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(680, 680)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[72.39239876, 82.90891754, 73.15835921],
                std=[1, 1, 1],
                to_rgb=True),
            dict(type='Pad', size=(680, 680), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='mydataset',
        data_root='/HOME/scz0aua/run/mmsegmentation-master/dataset',
        img_dir=
        '/HOME/scz0aua/run/mmsegmentation-master/dataset/data/data/images',
        ann_dir=
        '/HOME/scz0aua/run/mmsegmentation-master/dataset/data/data/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(680, 680),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[72.39239876, 82.90891754, 73.15835921],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='mydataset',
        data_root='/HOME/scz0aua/run/mmsegmentation-master/dataset',
        img_dir=
        '/HOME/scz0aua/run/mmsegmentation-master/dataset/data/data/images',
        ann_dir=
        '/HOME/scz0aua/run/mmsegmentation-master/dataset/data/data/masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(680, 680),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[72.39239876, 82.90891754, 73.15835921],
                        std=[1, 1, 1],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/HOME/scz0aua/run/mmsegmentation-master/checkpoints/cgnet_680x680_60k_cityscapes_20201101_110253-4c0b2f2d.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[1])
total_iters = 10000
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')
work_dir = 'work/mydataset'
gpu_ids = [0]
auto_resume = False
