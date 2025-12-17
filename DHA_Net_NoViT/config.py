config = {
    'seed': 42,
    'device': 'cuda',

    'data_root': '/mnt/data1/rove/dataset/S1_Water',
    'batch_size': 96,
    'num_workers': 12,

    'model_name': 'DHA_Net_NoViT',
    'in_channels': 2,
    'num_classes': 1,
    'use_bilinear': True,
    'image_size': 256,
    'patch_size': 16,

    'learning_rate': 5e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-3,
    'num_epochs': 300,
    'warmup_epochs': 10,

    'save_dir': './runs'
}
