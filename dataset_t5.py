import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import multiprocessing

class T5Dataset(Dataset):
    def __init__(self, folder_path):
        """
        初始化数据集
        :param folder_path: 包含.pt文件的文件夹路径
        """
        self.folder_path = folder_path
        # 获取所有.pt文件路径
        self.file_paths = glob.glob(os.path.join(folder_path, "*.pt"))
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {folder_path}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """加载单个样本"""
        file_path = self.file_paths[idx]
        try:
            # 加载数据
            data = torch.load(file_path)
            
            # 验证数据格式
            required_keys = {'prompt_embeds', 'pooled_prompt_embeds', 'prompt'}
            if not all(key in data for key in required_keys):
                missing = required_keys - set(data.keys())
                raise KeyError(f"Missing keys {missing} in file {file_path}")
            
            # 转换为张量（如果尚未是张量）
            prompt_embeds = data['prompt_embeds'].detach().requires_grad_(False)
            pooled_embeds = data['pooled_prompt_embeds'].detach().requires_grad_(False)
            
            # print(prompt_embeds.requires_grad)
            print('prompt_embeds:', prompt_embeds.shape)

            if not isinstance(prompt_embeds, torch.Tensor):
                prompt_embeds = torch.tensor(prompt_embeds, dtype=torch.float)
                
            if not isinstance(pooled_embeds, torch.Tensor):
                pooled_embeds = torch.tensor(pooled_embeds, dtype=torch.float)
                
            return {
                'prompt_embeds': prompt_embeds,
                'pooled_prompt_embeds': pooled_embeds,
                'prompt': data['prompt']  # 保持字符串类型
            }
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            # 返回空样本或使用错误处理策略
            return None

def collate_fn(batch):
    """处理可能的空样本并堆叠张量"""
    # 过滤掉None（错误样本）
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    print('ss:', [item['prompt_embeds'].shape for item in batch])
    # 堆叠张量数据
    prompt_embeds = torch.cat([item['prompt_embeds'] for item in batch], dim=0)
    pooled_prompt_embeds = torch.cat([item['pooled_prompt_embeds'] for item in batch], dim=0)

    
    # 保持文本列表
    prompts = [item['prompt'] for item in batch]
    
    print('prompt_embeds xx:', prompt_embeds.shape)

    return {
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'prompts': prompts
    }


def add_noise(x, noise, t1, t2):


    print('add_noise', t1, t2)
    sigma1 = t1 / 1000
    sigma2 = t2 / 1000

    sample = x * (1 - sigma2) / (1 - sigma1)

    print('sample', sample)
    beta = sigma2 ** 2 - (sigma1 * (1 - sigma2) / (1 - sigma1)) ** 2


    print('beta', beta)

    beta = beta ** 0.5

    return sample + beta * noise

# 使用示例
if __name__ == "__main__":
    # 初始化数据集和数据加载器
    # multiprocessing.set_start_method('spawn', force=True)

    # dataset = T5Dataset("t5_dataset")
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=4,  # 多进程加载
    #     collate_fn=collate_fn,
    #     pin_memory=False  # 加速GPU传输
    # )
    
    # # 迭代数据
    # for batch in dataloader:
    #     if batch is None:
    #         continue
            
    #     # 获取批次数据
    #     embeddings = batch['prompt_embeds']        # shape: [batch_size, ...]
    #     pooled = batch['pooled_prompt_embeds']     # shape: [batch_size, ...]
    #     texts = batch['prompts']                   # list of strings
        
    #     # 在这里使用数据...
    #     print(f"Batch embeddings shape: {embeddings.shape}")
    #     print(f"Pooled embeddings shape: {pooled.shape}")
    #     print(f"First text: {texts[0][:50]}...")

    #     break


    x1 = torch.ones((4, 512, 512), device='cuda:0')

    noise = torch.randn((4, 512, 512), device='cuda:0')

    

    t1 = torch.tensor([875.], device='cuda:0')
    t2 = torch.tensor([900], device='cuda:0')

    x2 = add_noise(x1, noise, t1, t1)


    print(x2)