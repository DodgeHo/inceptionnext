import os
import requests
import tarfile
import zipfile
import hashlib
from pathlib import Path
from tqdm import tqdm

DATASETS_CONFIG = {
    'coco_captions': {
        'urls': [
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            'http://images.cocodataset.org/zips/train2017.zip',
            'http://images.cocodataset.org/zips/val2017.zip'
        ],
        'md5': ['...', '...', '...']
    },
    'llava_instruct': {
        'url': 'https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json',
        'md5': '...'
    },
    'vqa_v2': {
        'urls': [
            'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip',
            'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip',
            'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip',
            'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip'
        ],
        'md5': ['...', '...', '...', '...']
    }
}

def download_file(url, dest):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(
            desc=f"Downloading {Path(dest).name}",
            total=total_size,
            unit='B',
            unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

if __name__ == '__main__':
    # 示例下载流程
    for dataset in ['coco_captions', 'llava_instruct', 'vqa_v2']:
        print(f'Processing {dataset}')
        config = DATASETS_CONFIG[dataset]
        
        # 创建数据集目录
        dataset_dir = Path(f'data/{dataset}')
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # MD5校验函数
        def verify_md5(file_path, expected_md5):
            hash_md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest() == expected_md5

        # 下载逻辑
        if 'urls' in config:
            for i, url in enumerate(config['urls']):
                filename = url.split('/')[-1]
                dest = dataset_dir / filename
                
                # 断点续传检查
                if dest.exists():
                    if verify_md5(dest, config['md5'][i]):
                        print(f'{filename} exists and MD5 verified')
                        continue
                    else:
                        print(f'{filename} MD5 mismatch, redownloading')
                        dest.unlink()
                
                download_file(url, dest)
                
                # 解压ZIP文件
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(dest, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
        
        # 单文件下载
        elif 'url' in config:
            filename = config['url'].split('/')[-1]
            dest = dataset_dir / filename
            
            if dest.exists():
                if verify_md5(dest, config['md5']):
                    print(f'{filename} exists and MD5 verified')
                    continue
                else:
                    print(f'{filename} MD5 mismatch, redownloading')
                    dest.unlink()
            
            download_file(config['url'], dest)