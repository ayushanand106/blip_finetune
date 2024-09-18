import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from PIL import Image
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from torch.cuda.amp import autocast, GradScaler

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    # Verify GPU availability
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Exiting.")
        return

    # Load model and processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16)
    model = model.to('cuda:0')

    # Freeze vision encoder and language model, unfreeze Q-Former
    for name, param in model.named_parameters():
        if "qformer" not in name:
            param.requires_grad = False

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=1,  # LoRA rank
        lora_alpha=8,  # Scaling factor
        lora_dropout=0.1,  # Dropout to avoid overfitting
        target_modules=["qformer.encoder"],
        
        # target_modules=["qformer.encoder.layer.*.attention"],  # Example pattern
        init_lora_weights="gaussian"  # Initialize LoRA weights with Gaussian distribution
    )

    # Apply LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print("LoRA applied. Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

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

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, num_workers=2)

    # Training parameters
    num_epochs = 10
    learning_rate = 1e-5
    warmup_steps = 100
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Training loop
    best_eval_loss = float('inf')
    accumulation_steps = 2  # Adjust based on memory

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            batch = {k: v.to('cuda:0') for k, v in batch.items()}
            labels = batch['input_ids'].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            with autocast():
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=labels
                )
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_train_loss += loss.item() * accumulation_steps

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
                batch = {k: v.to('cuda:0') for k, v in batch.items()}
                labels = batch['input_ids'].clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100

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
            'epoch': epoch+1,
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
