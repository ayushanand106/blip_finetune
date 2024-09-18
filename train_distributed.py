import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Blip2Processor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from PIL import Image
import subprocess

# Define the ImageCaptioningDataset class (unchanged)
class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, captions_file, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.image_captions = self.load_captions(captions_file)

    def load_captions(self, captions_file):
        image_captions = []
        with open(captions_file, 'r') as f:
            json_data = json.load(f)
            for img_name, captions in json_data.items():
                for caption in captions:
                    image_captions.append((img_name, caption))
        return image_captions

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        img_name, caption = self.image_captions[idx]
        image_path = os.path.join(self.image_dir, img_name + '.jpg')
        image = Image.open(image_path).convert('RGB')

        encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, return_tensors="pt")

        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        return encoding

class TrainDataset(ImageCaptioningDataset):
    def __init__(self, image_dir, captions_file, processor):
        super().__init__(image_dir, captions_file, processor)

class EvalDataset(ImageCaptioningDataset):
    def __init__(self, image_dir, captions_file, processor):
        super().__init__(image_dir, captions_file, processor)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_nvidia_smi():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    
def train(rank, world_size):
    setup(rank, world_size)
    
    # Load model and processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache/")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache/"
    ).to(rank)
    
    model = DDP(model, device_ids=[rank])

    # Freeze vision encoder and language model, unfreeze Q-Former
    for name, param in model.named_parameters():
        if "qformer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Setup dataloaders
    train_dataset = TrainDataset(
        image_dir='/leonardo_scratch/fast/EUHPC_E03_068/.cache/r2r/',
        captions_file='./train_annotations.json',
        processor=processor
    )
    eval_dataset = EvalDataset(
        image_dir='/leonardo_scratch/fast/EUHPC_E03_068/.cache/r2r/',
        captions_file='./val_annotations.json',
        processor=processor
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, sampler=eval_sampler, num_workers=4)

    num_epochs = 5
    learning_rate = 1e-4
    warmup_steps = 100
    gradient_accumulation_steps = 4
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    total_steps = len(train_dataloader) * num_epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_eval_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            batch = {k: v.to(rank) for k, v in batch.items()}
            labels = batch['input_ids'].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=labels
            )
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * gradient_accumulation_steps
            if rank == 0 and (step + 1) % 10 == 0:
                print(f"\nGPU status after iteration {step + 1}:")
                run_nvidia_smi()

        avg_train_loss = total_train_loss / len(train_dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        total_eval_loss = 0
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            batch = {k: v.to(rank) for k, v in batch.items()}
            labels = batch['input_ids'].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            with torch.no_grad():
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=labels
                )
                loss = outputs.loss

            total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1} - Average Evaluation Loss: {avg_eval_loss:.4f}")

            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_model.pth"))

    cleanup()

if __name__ == "__main__":
    world_size = 4  # Number of GPUs
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)