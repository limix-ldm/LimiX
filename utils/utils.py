import os
from huggingface_hub import snapshot_download, hf_hub_download

def download_datset(repo_id:str, revision:str, repo_type:str='dataset', save_dir:str="./my_cache"):
    print(f"Downloading {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
        local_dir=save_dir,
        ignore_patterns=None,
        force_download=False
    )
    
def list_folders_to_csv(path:str, output_csv:str):
    import csv
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['dataset name'])
        for folder in folders:
            writer.writerow([folder])
            
def download_model(repo_id:str, filename:str, save_path:str='.') -> str:
    file_path = hf_hub_download(
                                repo_id=repo_id,
                                filename=filename,
                                local_dir=save_path
                            )
    return file_path