config = {
    'seed': 3047,
    'device': 'cuda',

    'data_root': '/mnt/data1/rove/dataset/S1_Water',
    'batch_size': 16,
    'num_workers': 4,

    'model_name': 'MADF_Net',
    'in_channels': 2,
    'num_classes': 1,

    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-3,
    'num_epochs': 500,
    'warmup_epochs': 10,

    'save_dir': './runs'
}
