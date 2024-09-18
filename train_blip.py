import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from PIL import Image
from torch.cuda.amp import autocast
from peft import get_peft_model, LoraConfig
import math

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change to the appropriate GPU ID

# Define the ImageCaptioningDataset class
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

def main():
    # Load model and processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16
    ).to('cuda')
    
    # Freeze vision encoder and language model, unfreeze Q-Former
    for name, param in model.named_parameters():
        # param.requires_grad = False
    
        if "qformer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print(name)
    
    # model.eval()

    # Apply LoRA to the Q-Former module
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=32,  # Scaling factor
        # target_modules=["qformer.encoder.layer.10.attention.output.dense"], 
        # target_modules=["qformer"], 

        target_modules=["query",'key','value','dense'], 
        # target_modules=['q_proj', 'k_proj' ],
        # Targeting Q-Former layers
        lora_dropout=0.1,  # Dropout to avoid overfitting
        bias="none",
        # task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        # param.requires_grad = False
    
        if "qformer" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        print(name)
    
    model.print_trainable_parameters()

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Mixed-precision training
    scaler = torch.cuda.amp.GradScaler()


    # Setup dataloaders
    train_dataset = TrainDataset(
        image_dir='/media/mlr_lab/325C37DE7879ABF2/AyushAnand/r2r',
        captions_file='/media/mlr_lab/325C37DE7879ABF2/AyushAnand/train_annotations.json',
        processor=processor
    )
    eval_dataset = EvalDataset(
        image_dir='/media/mlr_lab/325C37DE7879ABF2/AyushAnand/r2r',
        captions_file='/media/mlr_lab/325C37DE7879ABF2/AyushAnand/val_annotations.json',
        processor=processor
    )



    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=6, num_workers=8)

    num_epochs = 5
    learning_rate = 1e-4
    warmup_steps = 100
    gradient_accumulation_steps = 16
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=AdamW(model.parameters(), lr=learning_rate),
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_eval_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            labels = batch['input_ids'].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            with autocast():
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_train_loss += loss.item() * gradient_accumulation_steps  # Multiply back

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        total_eval_loss = 0
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            labels = batch['input_ids'].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            with torch.no_grad():
                with autocast():
                    outputs = model(
                        pixel_values=batch['pixel_values'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=labels
                    )
                    loss = outputs.loss

            total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch+1} - Average Evaluation Loss: {avg_eval_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'eval_loss': avg_eval_loss
        }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        print(f"Checkpoint saved: {os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')}")

        # Save best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"New best model saved: {os.path.join(save_dir, 'best_model.pth')}")

    print("Training completed!")

if __name__ == "__main__":
    main()
