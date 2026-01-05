"""
Configuration for MoR Benchmarking Experiments
"""

class Config:
    # Model Architecture
    d_model = 256
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 64
    
    # Training
    batch_size = 64
    learning_rate = 1e-3
    epochs_baseline = 30
    epochs_mor_exp1 = 30
    epochs_mor_exp2 = 50
    
    # MoR Specific
    router_hidden_dim = 128
    gumbel_temperature = 1.0
    lambda_depth_penalty = 0.1  # Auxiliary loss weight for depth
    
    # Experiment Configurations
    n_layers_deep = 12
    n_layers_shallow = 6
    
    # Dataset
    vocab_size_shakespeare = 55  # Will be set dynamically
    vocab_size_wikitext = None   # Will be set dynamically
    
    # Device
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    # Paths
    data_dir = './data'
    checkpoint_dir = './checkpoints'
    results_dir = './results'
