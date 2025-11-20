config = {
    'seed': 3047,
    'device': 'cuda',

    'data_root': '/mnt/data1/rove/dataset/S1_Water',
    'batch_size': 48,
    'num_workers': 4,

    'model_name': 'Nested_UNet',
    'in_channels': 3,
    'num_classes': 1,
    'use_bilinear': True,
    'image_size': 256,
    'patch_size': 16,

    'embed_dim': 768,   
    'depth': 3,            
    'num_heads': 6,
    'mlp_ratio': 4.0,
    'align_lambda': 0.0,
    'cls_lambda': 0.0,

    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-3,
    'num_epochs': 500,
    'warmup_epochs': 10,

    'save_dir': './runs'
}
