config = {
    'seed': 3047,
    'device': 'cuda',

    'data_root': '/mnt/data1/rove/dataset/S1_Water',
    'batch_size': 96,
    'num_workers': 8,

    'model_name': 'GCAFF_Net',
    'in_channels': 2,
    'num_classes': 1,
    'image_size': 256,

    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-3,
    'num_epochs': 300,
    'warmup_epochs': 10,

    'save_dir': './runs'
}
