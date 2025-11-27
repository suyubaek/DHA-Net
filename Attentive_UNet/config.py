config = {
    'seed': 3047,
    'device': 'cuda',

    'data_root': '/mnt/data1/rove/dataset/S1_Water_512',
    'batch_size': 8,
    'num_workers': 4,

    'model_name': 'Attentive_UNet',
    'in_channels': 2,
    'num_classes': 1,
    'use_bilinear': True,
    'image_size': 512,

    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-3,
    'num_epochs': 300,
    'warmup_epochs': 10,

    'save_dir': './runs'
}
