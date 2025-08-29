import torch

from model.transformer import FeaturesTransformer


def load_model(model_path,calculate_sample_attention:bool=False,calculate_feature_attention:bool=False,mask_prediction:bool=False):
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config = state_dict['config']
    model = FeaturesTransformer(
        preprocess_config_x=config['preprocess_config_x'],
        encoder_config_x=config['encoder_config_x'],
        encoder_config_y=config['encoder_config_y'],
        decoder_config=config['decoder_config'],
        nlayers=config['nlayers'],
        nhead=config['nhead'],
        embed_dim=config['embed_dim'],
        hid_dim=config['hid_dim'],
        mask_prediction=mask_prediction,
        features_per_group=config['features_per_group'],
        dropout=config['dropout'],
        layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        device=None,
        dtype=None,
        recompute_attn=config['recompute_attn'],
        calculate_sample_attention=calculate_sample_attention,
        calculate_feature_attention=calculate_feature_attention
    )
    model.load_state_dict(state_dict['state_dict'])

    model.eval()
    return model