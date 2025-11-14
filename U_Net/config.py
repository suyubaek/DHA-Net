config = {
    'seed': 42,
    'device': 'cuda',

    'data_root': '/mnt/sda1/songyufei/dataset/lancang',
    'batch_size': 4,
    'num_workers': 4,

    'model_name': 'U_Net',
    'sar_channels': 1,
    'num_classes': 1,
    'use_bilinear': False,
    'image_size': 256,

    'learning_rate': 5e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-3,
    'num_epochs': 150,
    'warmup_epochs': 5,

    
    'save_dir': './runs'
}
