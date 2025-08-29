import torch
import torch.nn as nn
from model.layer import EncoderBaseLayer, MLP
from typing import Any,Literal
from torch.nn.init import orthogonal_
import numpy as np

def calc_mean(x:torch.Tensor, dim:int):
    num = torch.sum(~torch.isnan(x), dim=dim).clip(min=1.0)
    return torch.nansum(x, dim=dim) / num, num

def calc_std(x:torch.Tensor, dim:int, mean_v:torch.Tensor|None = None, value_num:torch.Tensor|None=None ):
    if mean_v is None or value_num is None:
        mean_v, value_num = calc_mean(x, dim)
    mean_broadcast = torch.repeat_interleave(mean_v.unsqueeze(dim), x.shape[dim], dim=dim,)
    return torch.sqrt(torch.nansum(torch.square(mean_broadcast - x), dim=dim) / (value_num - 1))

def drop_outliers(
                    x:torch.Tensor, 
                    std_sigma:float=4,
                    eval_pos:int=-1,
                    lower:torch.Tensor|None = None,
                    upper:torch.Tensor|None = None,
                    dim:int=1
                    ):
        assert len(x.shape)==3, "x.shape must be B,S,F"

        if lower is None:
            data = x[:,:eval_pos].clone()
            data_mean, value_num = calc_mean(data, dim=dim)
            data_std = calc_std(data, dim=dim, mean_v=data_mean, value_num=value_num)
            cut_off = data_std * std_sigma
            lower, upper = data_mean - cut_off, data_mean + cut_off
            
            data[torch.logical_or(data > upper, data < lower)] = np.nan
            data_mean, value_num = calc_mean(data, dim=dim)
            data_std = calc_std(data, dim=dim, mean_v=data_mean, value_num=value_num)
            cut_off = data_std * std_sigma
            lower, upper = data_mean - cut_off, data_mean + cut_off
        
        x = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
        x = torch.minimum(torch.log(1 + torch.abs(x)) + upper, x)
        
        return x, lower, upper
    
def normalize_mean0_std1(
                        x:torch.Tensor, 
                        eval_pos:int=-1,
                        clip:bool=True,
                        dim:int=1,
                        mean: torch.Tensor | None = None,
                        std: torch.Tensor | None = None
                        ):
    if mean is None:
        mean, value_num = calc_mean(x[:,:eval_pos], dim=dim)
        std = calc_std(x[:,:eval_pos], dim=dim, mean_v=mean, value_num=value_num) + 1e-20
        
        if x.shape[1] == 1 or eval_pos == 1:
            std[:] = 1.0
    x = (x - mean.unsqueeze(1).expand_as(x)) / std.unsqueeze(1).expand_as(x)
    if clip:
        x = torch.clip(x, min=-100, max=100)
    return x, mean, std
    

class LinearEncoder(nn.Module):
    """linear input encoder"""
    def __init__(
                self,
                num_features: int,
                emsize: int,
                nan_to_zero: bool = False,
                bias: bool = True,
                in_keys:list[str]=['data'],
                out_key:str='data',
    ):
        """Initialize the LinearEncoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.nan_to_zero = nan_to_zero
        self.in_keys = in_keys
        self.out_key = out_key
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        assert 'data' in input and 'nan_encoding' in input
        x = [input[key] for key in self.in_keys] 
        x = torch.cat(x, dim=-1) # type: ignore
        if self.nan_to_zero:
            x = torch.nan_to_num(x, nan=0.0)
            
        input[self.out_key] = self.layer(x)
        return input

class MLPEncoder(nn.Module):
    """MLP input encoder"""
    def __init__(
                self,
                num_features: int,
                emsize: int,
                nan_to_zero: bool = False,
                bias: bool = True,
                in_keys: list[str] = ['data'],
                out_key: str = 'data',
    ):
        """Initialize the MLPEncoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_features, emsize * 2, bias=bias),
            nn.LayerNorm(emsize * 2),
            nn.GELU(),
            nn.Linear(emsize * 2, emsize, bias=bias),
            nn.LayerNorm(emsize),
            nn.GELU()
        )
        self.nan_to_zero = nan_to_zero
        self.in_keys = in_keys
        self.out_key = out_key
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        assert 'data' in input and 'nan_encoding' in input
        x = [input[key] for key in self.in_keys]
        x = torch.cat(x, dim=-1) # type: ignore
        if self.nan_to_zero:
            x = torch.nan_to_num(x, nan=0.0)
        input[self.out_key] = x
        return input

class MaskEmbEncoder(nn.Module):
    """
    For masked features, use the mask vector to obtain their representations; 
    for numerical features, use a nonlinear network to obtain their representations
    """
    def __init__(
                self,
                num_features: int,
                emsize: int,
                mask_embedding_size: int,
                nan_to_zero: bool = False,
                bias: bool = True,
                in_keys: list[str] = ['data'],
                out_key: str = 'data',
    ):
        """Initialize the MaskEmbEncoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.embedding_dim = emsize
        self.mask_embedding_size = mask_embedding_size
        self.in_keys = in_keys
        self.out_key = out_key

        # All masked positions use the same vector
        self.mask_embedding = nn.Parameter(torch.randn(self.mask_embedding_size))

        # MLP for numerical features: input is 1, output is embedding_dim
        self.numeric_mlp = nn.Sequential(
            nn.Linear(1, self.embedding_dim // 2, bias=bias),
            nn.LayerNorm(self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim, bias=bias),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU()
        )

        # Merging layer: maps the concatenated feature vectors back to embedding_dim.
        self.fusion_network = nn.Sequential(
            nn.Linear(num_features * self.embedding_dim, self.embedding_dim, bias=bias),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias),
            nn.LayerNorm(self.embedding_dim)
        )
        self.nan_to_zero = nan_to_zero
    
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        assert 'data' in input and 'nan_encoding' in input
        x = [input[key] for key in self.in_keys]
        x = torch.cat(x, dim=-1) # type: ignore
        batch_size, seq_len, group, feature_num = x.shape
        x_flat = x.view(-1, feature_num)
        is_mask = torch.isnan(x_flat)
        feature_embeddings = []   
        for i in range(feature_num):
            feat_vals = x_flat[:, i].unsqueeze(-1)
            feat_is_mask = is_mask[:, i].unsqueeze(-1)

            # Processing numerical features
            numeric_input = torch.where(~feat_is_mask, feat_vals, torch.zeros_like(feat_vals))
            numeric_emb = self.numeric_mlp(numeric_input)

            # Construct mask embedding
            mask_emb = self.mask_embedding.expand(numeric_emb.shape[0], -1)
            
            # Merge the embedding results of masked features and numerical features
            combined_emb = torch.where(feat_is_mask.expand_as(numeric_emb), mask_emb, numeric_emb)
            feature_embeddings.append(combined_emb)
        concat_vector = torch.cat(feature_embeddings, dim=-1)

        sample_representation = self.fusion_network(concat_vector)
        output = sample_representation.view(batch_size, seq_len, group, -1)
        
        
        input[self.out_key] = output
        return input

class NanEncoder(nn.Module):
    """Encoder stage that deals with NaN and infinite values in the input"""
    def __init__(
        self,
        nan_value: float = -2.0,
        inf_value: float = 2.0,
        neg_info_value: float = 4.0,
        in_keys:list[str]=['data'],
        out_key:str='nan_encoding'
    ):
        """Initialize the NanEncoder.

        Args:
            keep_nans: Flag to maintain NaN values as individual indicators. 
        """
        super().__init__()
        self.nan_value = nan_value
        self.inf_value = inf_value
        self.neg_info_value = neg_info_value
        self.in_keys = in_keys
        self.out_key = out_key
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x:torch.Tensor = input[self.in_keys[0]] # type: ignore
        eval_pos = input['eval_pos']
        
        mean_value, _ = calc_mean(x[:,:eval_pos,:], dim=1)
        
        nans_indicator = torch.zeros_like(x, dtype=x.dtype)
        nans_indicator[torch.isnan(x)] = self.nan_value
        pos_inf_mask = torch.isinf(x) & (torch.sign(x) == 1)
        nans_indicator[pos_inf_mask] = self.inf_value
        neg_inf_mask = torch.isinf(x) & (torch.sign(x) == -1)
        nans_indicator[neg_inf_mask] = self.neg_info_value
        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        # avoid inplace operations
        x = x.clone()
        x[nan_mask] = mean_value.unsqueeze(1).expand_as(x)[nan_mask]
        
        input[self.in_keys[0]] = x
        input[self.out_key ] = nans_indicator
        return input
        
    
class ValidFeatureEncoder(nn.Module):
    """Valid feature encoder"""
    def __init__(
        self,
        num_features: int,
        nan_normalize: bool=True,
        sqrt_normalize: bool=True,
        in_keys:list[str]=['data'],
        out_key:str='data'
    ):
        """Initialize the ValidFeatureEncoder.

        Args:
            num_features: The target number of features to transform the input into.
            nan_normalize: Indicates whether to normalize based on the number of features actually used.
            sqrt_normalize: Legacy option to normalize using the square root rather than the count of used features.
        """
        super().__init__()
        self.num_features = num_features
        self.nan_normalize = nan_normalize
        self.sqrt_normalize = sqrt_normalize
        self.in_keys = in_keys
        self.out_key = out_key
        self.valid_feature_num = None
    
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x:torch.Tensor = input[self.in_keys[0]]  # type: ignore
        valid_feature = ~torch.all(x == x[:, 0:1, :], dim=1)
        self.valid_feature_num = torch.clip(valid_feature.sum(-1).unsqueeze(-1),min=1)

        if self.nan_normalize:
            if self.sqrt_normalize:
                x = x * torch.sqrt(self.num_features / self.valid_feature_num).unsqueeze(1).expand_as(x)
            else:
                x = x * (self.num_features / self.valid_feature_num)
        
        zeros = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros], -1)
        
        input[self.out_key] = x
        return input
    

class EmbYEncoderStep(nn.Module):
    """A simple linear input encoder step."""

    def __init__(
        self,
        *,
        emsize: int,
        n_classes: int = 10,
        in_keys: list[str] = ['data'],
        out_key: str = 'data',
    ):
        """Initialize the EmbYEncoderStep.

        Args:
            emsize: The embedding size, i.e. the number of output features.
            n_classes: Number of classes
        """
        super().__init__()
        
        # Ensure the embedding dimension is large enough to support orthogonal initialization.
        assert emsize > n_classes + 1, (f"emsize ({emsize}) must be >= n_classes+1 ({n_classes+1}) for orthogonal initialization")

        # Generate an orthogonal matrix of size (n_classes + 1) × emsize
        ortho_matrix = torch.empty(n_classes + 1, emsize)
        orthogonal_(ortho_matrix)  # Initialize in-place as an orthogonal matrix

        # Decompose the matrix: the first n_classes rows are used for y_embedding, and the last row is used for y_mask.
        y_embed_weights = ortho_matrix[:n_classes, :]  # Shape (n_classes, emsize)
        y_mask_weight = ortho_matrix[n_classes:n_classes+1, :]  # Shape (1, emsize)

        self.y_embedding = nn.Embedding(n_classes, emsize)
        self.y_embedding.weight.data = y_embed_weights.clone()

        self.y_mask = nn.Embedding(1, emsize)
        self.y_mask.weight.data = y_mask_weight.clone()
        self.in_keys = in_keys
        self.out_key = out_key
        if len(self.in_keys) > 1:
            print("Warning: The EmbYEncoderStepl function is only for processing Y, and in_keys must contain exactly one key.")
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        y = input[self.in_keys[0]]
        eval_pos = input['eval_pos']
        y = y.int() # type: ignore
        y_train = y[:,:eval_pos]
        y_test = torch.zeros_like(y[:, eval_pos:], dtype=torch.int)
        y_train_emb = self.y_embedding(y_train).to(torch.float16)
        y_test_emb = self.y_mask(y_test).to(torch.float16)
        y_emb = torch.cat([y_train_emb, y_test_emb], dim=1)
        
        input[self.out_key] = y_emb
        return input

class MulticlassTargetEncoder(nn.Module):
    """Use the target's index as the class value, with each class corresponding to an index"""
    def __init__(
        self,
        in_keys:list[str]=['data'],
        out_key:str='data'
    ):
        """Initialize the ValidFeatureEncoder.

        Args:
            in_keys: the keys of the input parameter
            out_key: the key of the output result.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_key = out_key
    
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x:torch.Tensor = input[self.in_keys[0]]  # type: ignore
        eval_pos = input['eval_pos']
        unique_xs = [
            torch.unique(x[b, :eval_pos]) for b in range(x.shape[0])
        ]
        x_ = x.clone()
        for b in range(x.shape[0]):
            x_[b, :, :] = (x[b, :, :].unsqueeze(-1) > unique_xs[b]).sum(dim=-1)
   
        input[self.out_key] = x_
        return input

class NormalizationEncoder(nn.Module):
    """normalize encoder"""
    def __init__(
                self, 
                train_only:bool,
                normalize_x:bool,
                remove_outliers:bool,
                std_sigma:float=4.0,
                in_keys:list[str]=['data'],
                out_key:str='data'
                
    ):
        super().__init__()
        self.train_only = train_only
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.std_sigma = std_sigma
        self.in_keys = in_keys
        self.out_key = out_key
        self.mean = None
        self.std = None

    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x = input[self.in_keys[0]]
        eval_pos = input['eval_pos']
        pos = eval_pos if self.train_only else -1
        if self.remove_outliers:
            x, lower, upper = drop_outliers(x, eval_pos=pos, std_sigma=self.std_sigma)
        if self.normalize_x:
            x, self.mean, self.std = normalize_mean0_std1(x, eval_pos=pos )
        
        input[self.out_key] = x
        return input



def get_x_encoder(
    *,
    num_features: int,
    embedding_size: int,
    mask_embedding_size: int,
    encoder_use_bias: bool,
    in_keys: list = ['data']
):
    inputs_to_merge = {}
    for in_key in in_keys:
        inputs_to_merge[in_key] = {'dim': num_features}

    encoder_steps = [
        MaskEmbEncoder(
            num_features=sum([i["dim"] for i in inputs_to_merge.values()]),
            emsize=embedding_size,
            mask_embedding_size=mask_embedding_size,
            bias=encoder_use_bias,
        ),
    ]

    return nn.Sequential(*encoder_steps,)


def get_cls_y_encoder(
    *,
    num_inputs: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int
) -> nn.Module:
    steps = []
    inputs_to_merge = [{"name": "data", "dim": num_inputs}]
    if nan_handling_y_encoder:
        steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]
        inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]

    if max_num_classes >= 2:
        steps += [MulticlassTargetEncoder()]

    steps += [
            EmbYEncoderStep(
                emsize=embedding_size,
                n_classes=max_num_classes
        )
    ]
    return nn.Sequential(*steps)

def get_reg_y_encoder(
    *,
    num_inputs: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int
) -> nn.Module:
    steps = []
    inputs_to_merge = [{"name": "data", "dim": num_inputs}]
    if nan_handling_y_encoder:
        steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]
        inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]

    steps += [
        LinearEncoder(
            num_features=sum([i["dim"] for i in inputs_to_merge]),  # type: ignore
            emsize=embedding_size,
            in_keys=['data', 'nan_encoding'],
            out_key='data'
        ),
    ]
    return nn.Sequential(*steps)


def preprocesss_4_x(
    *,
    num_features: int,
    nan_handling_enabled: bool,
    normalize_on_train_only: bool,
    normalize_x: bool,
    remove_outliers: bool,
    normalize_by_used_features: bool,
    ):
    """feature preprocess"""
    inputs_to_merge = {"data": {"dim": num_features}}

    preprocess_steps = []

    # Obtain the positions of features with NaN and Inf values, and replace these features with the mean of the corresponding feature
    preprocess_steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]   
    
    if nan_handling_enabled:
        inputs_to_merge["nan_encoding"] = {"dim": num_features}
        preprocess_steps += [
            # Zero values are added to convert the input into a fixed number of features, without normalization (variance is not constant). 
            # This transformation is applied to the nan_indicators set, which shares the same shape as x. 
            # However, since x has been imputed prior to this step, this operation is theoretically redundant.
            ValidFeatureEncoder(
                num_features=num_features,
                nan_normalize=False,
                in_keys=["nan_encoding"],
                out_key="nan_encoding"
            ),
        ]

    preprocess_steps += [
        NormalizationEncoder(
            train_only=normalize_on_train_only,
            normalize_x=normalize_x,
            remove_outliers=remove_outliers,
        ),
    ]

    preprocess_steps += [
        # Convert the input into a fixed number of features by adding zero values, with normalization applied (variance is constant).
        ValidFeatureEncoder(
            num_features=num_features,
            nan_normalize=normalize_by_used_features,
        ),
    ]

    return nn.Sequential(*preprocess_steps)