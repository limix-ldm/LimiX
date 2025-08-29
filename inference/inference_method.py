import gc
from typing import Literal, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from utils.data_utils import TabularInferenceDataset
from utils.inference_utils import NonPaddingDistributedSampler, swap_rows_back
from utils.loading import load_model

from utils.retrieval_utils import RelabelRetrievalY
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():

    if dist.is_initialized():
        return dist.get_rank(),dist.get_world_size()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup():
    if not dist.is_initialized():
        print("Distributed environment is not initialized, nothing to clean up.")
        return

    print("Cleaning up distributed environment...")
    dist.destroy_process_group()
    print("Distributed environment cleaned up.")

class InferenceResultWithRetrieval:
    def __init__(self,
                 model: torch.nn.Module,
                 sample_selection_type: Literal["AM", "DDP"] = "AM",
                 ):
        self.model=model
        self.sample_selection_type = sample_selection_type
        self.dataset = None

    def _prepare_data(self,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      attention_score: np.ndarray = None,
                      retrieval_len: int = 2000
                      ) -> TabularInferenceDataset:
        if self.sample_selection_type == "AM":
            use_retrieval = True
        else:
            use_retrieval = False
        dataset = TabularInferenceDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            attention_score=attention_score,
            retrieval_len=retrieval_len,
            use_retrieval=use_retrieval
        )
        return dataset

    def inference(self,
                  X_train: torch.Tensor = None,
                  y_train: torch.Tensor = None,
                  X_test: torch.Tensor = None,
                  dataset: TabularInferenceDataset = None,
                  attention_score: np.ndarray | torch.Tensor = None,
                  retrieval_len: int = 2000,
                  dynamic_ratio:float=None,
                  task_type: Literal["reg", "cls"] = "reg"):
        self.rank,self.world_size = setup()
        model = self.model.cuda(self.rank)
        model = DDP(model, device_ids=[self.rank],find_unused_parameters=False)
        if isinstance(retrieval_len,str):
            if retrieval_len == "dynamic":
                if dynamic_ratio is not None:
                    retrieval_len =int(dynamic_ratio*X_train.shape[0]/len(torch.unique(y_train)))
                else:
                    retrieval_len = int(X_train.shape[0]/len(torch.unique(y_train)))
        if isinstance(retrieval_len, float):
            self.retrieval_len = int(retrieval_len * X_train.shape[0])
        else:
            self.retrieval_len = retrieval_len
        if dataset is None:
            dataset = self._prepare_data(X_train, y_train, X_test, attention_score, self.retrieval_len)
        sampler = NonPaddingDistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        outputs = []
        dataloader = DataLoader(dataset,
                                batch_size=16,
                                shuffle=False,
                                drop_last=False,
                                sampler=sampler
                                )
        indice = []
        for data in dataloader:
            with (
                torch.autocast(torch.device(model.device).type, enabled=True),
                torch.inference_mode(),
            ):
                if self.sample_selection_type == "DDP":
                    indice.append(data["idx"])
                    X_test = data["X_test"].unsqueeze(1)
                    X_train_item = torch.cat([X_train.unsqueeze(0) for _ in range(X_test.shape[0])], dim=0)
                    Y_train_item = torch.cat([y_train.unsqueeze(0).unsqueeze(-1) for _ in range(X_test.shape[0])], dim=0)
                    x_ = torch.cat([X_train_item, X_test], dim=1)
                    output = model(x=x_, y=Y_train_item.squeeze(-1), eval_pos=Y_train_item.shape[1], task_type=task_type)
                else:
                    indice.append(data["idx"])
                    X_train = data["X_train"]
                    X_test = data["X_test"].unsqueeze(1)
                    y_ = data["y_train"]
                    x_ = torch.cat([X_train, X_test], dim=1)
                    if task_type == "cls":
                        relabel = RelabelRetrievalY(y_)
                        y_ = relabel.transform_y()
                    output=model(x=x_, y=y_.squeeze(-1), eval_pos=y_.shape[1], task_type=task_type)
                    if len(output.shape) == 3:
                        output = output.view(-1, output.shape[-1])
                    if task_type == "cls":
                        output = output.cpu().numpy()
                        output = relabel.inverse_transform_y(output)
                        output = torch.tensor(output, dtype=torch.float32, device=model.device)

            outputs.append(output.cpu())
            del output
            gc.collect()
            torch.cuda.empty_cache()
        del model
        outputs = torch.cat(outputs, dim=0)
        local_result_cpu = outputs.cpu()
        indice = torch.cat(indice, dim=0)
        local_indice_cpu = indice.cpu()
        outputs = [None for _ in range(self.world_size)]
        gathered_indice = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_indice, local_indice_cpu)
        dist.all_gather_object(outputs, local_result_cpu)
        del local_result_cpu
        outputs = torch.cat(outputs, dim=0).to(torch.float32)
        gathered_indice = torch.cat(gathered_indice, dim=0)
        outputs = swap_rows_back(outputs, gathered_indice)
        gc.collect()
        torch.cuda.empty_cache()
        return outputs.squeeze(0)


class InferenceAttentionMap:
    def __init__(self,
                 model_path: str,
                 calculate_feature_attention: bool = False,
                 calculate_sample_attention: bool = False,
                 ):
        self.calculate_feature_attention = calculate_feature_attention
        self.calculate_sample_attention = calculate_sample_attention
        self.model = load_model(model_path, calculate_feature_attention=calculate_feature_attention,
                           calculate_sample_attention=calculate_sample_attention)

        self.dataset = None

    def _prepare_data(self,
                      X_train: torch.Tensor,
                      y_train: torch.Tensor,
                      X_test: torch.Tensor,
                      ) -> TabularInferenceDataset:
        dataset = TabularInferenceDataset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            use_retrieval=False
        )
        return dataset

    def inference(self,
                  X_train: torch.Tensor | np.ndarray,
                  y_train: torch.Tensor | np.ndarray,
                  X_test: torch.Tensor | np.ndarray,
                  task_type: Literal["reg", "cls"] = "reg") -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self.rank, self.world_size = setup()
        # device = torch.device(f"cuda:{self.rank}")
        model = self.model.cuda(self.rank)
        model = DDP(model, device_ids=[self.rank])
        model.eval()
        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train).float()
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float()
        dataset = self._prepare_data(X_train, y_train, X_test)

        sampler = NonPaddingDistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        dataloader = DataLoader(dataset,
                                batch_size=16,
                                shuffle=False,
                                drop_last=False,
                                sampler=sampler
                                )
        local_feature_attention = []
        local_sample_attention = []
        feature_attention=None
        sample_attention=None
        indice=[]
        for batch_idx, data in enumerate(dataloader):
            X_test=data["X_test"]
            idx=data["idx"]
            indice.append(idx)
            x_=torch.cat([X_train,X_test],dim=0).unsqueeze(dim=0)

            y_=y_train.unsqueeze(0)
            with(torch.autocast(device_type='cuda', enabled=True), torch.inference_mode()):
                output,feature_attention,sample_attention = model(x=x_, y=y_, eval_pos=y_.shape[1], task_type=task_type)

            if self.calculate_sample_attention:
                local_sample_attention.append(sample_attention.permute(1,0,2))
            if self.calculate_feature_attention:
                local_feature_attention.append(feature_attention[y_.shape[1]:,:,:])
            del output, sample_attention, feature_attention,X_test
            gc.collect()
            torch.cuda.empty_cache()
        indice=torch.cat(indice,dim=0)
        if self.calculate_feature_attention:
            feature_attentions = torch.cat(local_feature_attention,dim=0)# shape[len_Dtest, feature_num//feature_per_group, feature_num//feature_per_group,]
            local_result_cpu = feature_attentions.cpu()
            local_indice_cpu=indice.cpu()
            gathered_feature = [None for _ in range(self.world_size)]
            gathered_indice = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_feature, local_result_cpu)
            dist.all_gather_object(gathered_indice, local_indice_cpu)
            feature_attention = torch.cat(gathered_feature, dim=0)
            gathered_indice=torch.cat(gathered_indice, dim=0)
            feature_attention=swap_rows_back(feature_attention, gathered_indice)
            del gathered_feature
        if self.calculate_sample_attention:
            sample_attentions = torch.cat(local_sample_attention,dim=0)
            local_indice_cpu = indice.cpu()
            local_result_cpu = sample_attentions.cpu()
            gathered_sample = [None for _ in range(self.world_size)]
            gathered_indice = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_sample, local_result_cpu)
            dist.all_gather_object(gathered_indice, local_indice_cpu)
            sample_attention = torch.cat(gathered_sample, dim=0)
            gathered_indice = torch.cat(gathered_indice, dim=0)
            sample_attention = swap_rows_back(sample_attention, gathered_indice)
            del gathered_sample


        dist.barrier()
        del sample_attentions,feature_attentions,model
        gc.collect()
        torch.cuda.empty_cache()
        return feature_attention, sample_attention.permute(1,0,2)
