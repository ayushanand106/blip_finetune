import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from PIL import Image
import torch
from torch.cuda.amp import autocast, GradScaler

# Set CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

        # Adjust padding and truncation with a set max_length
        encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        # Add decoder_input_ids
        decoder_input_ids = self.processor.tokenizer.build_inputs_with_special_tokens(encoding['input_ids'].tolist())
        encoding['decoder_input_ids'] = torch.tensor(decoder_input_ids)

        return encoding


def collate_fn(batch):
    # Stack pixel_values and apply padding to input_ids, attention_mask, and decoder_input_ids
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    decoder_input_ids = torch.nn.utils.rnn.pad_sequence([item['decoder_input_ids'] for item in batch], batch_first=True)

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids
    }

class TrainDataset(ImageCaptioningDataset):
    def __init__(self, image_dir, captions_file, processor):
        super().__init__(image_dir, captions_file, processor)

class EvalDataset(ImageCaptioningDataset):
    def __init__(self, image_dir, captions_file, processor):
        super().__init__(image_dir, captions_file, processor)

def main():
    # Load model and processor
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="cuda:0", torch_dtype=torch.float16).to(device)

    # Freeze vision encoder and language model, unfreeze Q-Former
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False
    for param in model.qformer.parameters():
        param.requires_grad = True

    # Setup dataloaders
    train_dataset = TrainDataset(
        image_dir='/media/mlr_lab/325C37DE7879ABF2/AyushAnand/r2r',
        captions_file='/media/mlr_lab/325C37DE7879ABF2/AyushAnand/test.json',
        processor=processor
    )
    eval_dataset = EvalDataset(
        image_dir='/media/mlr_lab/325C37DE7879ABF2/AyushAnand/r2r',
        captions_file='/media/mlr_lab/325C37DE7879ABF2/AyushAnand/test.json',
        processor=processor
    )

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, collate_fn=collate_fn)

    # Training parameters
    num_epochs = 10
    learning_rate = 5e-5
    warmup_steps = 100
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Use GradScaler for mixed precision
    scaler = GradScaler()

    # Training loop
    best_eval_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            # Move inputs to the correct device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            
            # Forward pass with autocast for mixed precision
            with autocast():
                outputs = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['decoder_input_ids'],
                )
                loss = outputs.loss
            
            # Scales the loss, calls backward and unscales gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        total_eval_loss = 0
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            with torch.no_grad():
                # Move inputs to the correct device
                batch = {k: v.to(device) for k, v in batch.items()}

                with autocast():
                    outputs = model(
                        pixel_values=batch['pixel_values'],
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        decoder_input_ids=batch['decoder_input_ids'],
                        labels=batch['decoder_input_ids']  # Use decoder_input_ids as labels
                    )
                    loss = outputs['loss']
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