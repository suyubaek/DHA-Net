config = {
    'seed': 42,
    'device': 'cuda',

    'data_root': '/mnt/sda1/songyufei/dataset/lancang',
    'batch_size': 8,
    'num_workers': 4,

    'model_name': 'Hybrid TransCNN',
    'in_channels': 1,
    'num_classes': 1,
    'use_bilinear': True,
    'image_size': 256,
    'patch_size': 16,

    'embed_dim': 768,   
    'depth': 4,            
    'num_heads': 6,
    'mlp_ratio': 4.0,

    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-3,
    'num_epochs': 200,
    'warmup_epochs': 5,

    
    'save_dir': './runs'
}
