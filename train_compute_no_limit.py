import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from PIL import Image
from accelerate import Accelerator

# Initialize Accelerator
accelerator = Accelerator()
device = accelerator.device

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
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache/")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache/", device_map="auto"
    ).to(device)  # Move model to the correct device

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

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, num_workers=8)

    num_epochs = 5
    learning_rate = 1e-4
    warmup_steps = 100
    gradient_accumulation_steps = 16
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    total_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Prepare everything with Accelerator (change)
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    best_eval_loss = float('inf')
    
    with accelerator.main_process_first():  # Use Accelerator's context manager (change)
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
                batch = {k: v.to(device) for k, v in batch.items()}  # Use device from Accelerator (change)
                labels = batch['input_ids'].clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100

                with accelerator.autocast():  # Use accelerator's autocast (change)
                    outputs = model(
                        pixel_values=batch['pixel_values'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps

                accelerator.backward(loss)  # Use accelerator's backward (change)

                if (step + 1) % gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Optional: clip gradients (change)
                    accelerator.step(optimizer)  # Use accelerator's step (change)
                    scheduler.step()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * gradient_accumulation_steps  # Multiply back

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

            # Evaluation
            model.eval()
            total_eval_loss = 0
            for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
                batch = {k: v.to(device) for k, v in batch.items()}  # Use device from Accelerator (change)
                labels = batch['input_ids'].clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100

                with torch.no_grad():
                    with accelerator.autocast():  # Use accelerator's autocast (change)
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

            # Save checkpoint and best model logic remains unchanged

if __name__ == "__main__":
    main()
