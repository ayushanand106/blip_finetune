import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW, SGD
from tqdm import tqdm
import os
import json
from PIL import Image
# from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# non_t_params=['qformer.encoder.layer.1.attention.output.dense.bias', 'qformer.encoder.layer.7.intermediate_query.dense.weight', 'qformer.encoder.layer.8.attention.output.LayerNorm.bias', 'qformer.encoder.layer.6.crossattention.attention.query.weight', 'qformer.encoder.layer.8.crossattention.output.LayerNorm.bias', 'qformer.encoder.layer.3.output_query.dense.bias', 'qformer.encoder.layer.0.crossattention.output.dense.bias', 'qformer.encoder.layer.7.attention.output.dense.bias', 'qformer.encoder.layer.7.output_query.LayerNorm.bias', 'qformer.encoder.layer.4.attention.attention.value.weight', 'qformer.encoder.layer.4.intermediate_query.dense.weight', 'qformer.encoder.layer.1.attention.attention.value.weight', 'qformer.encoder.layer.2.crossattention.attention.key.weight', 'qformer.encoder.layer.3.output_query.dense.weight', 'qformer.encoder.layer.9.attention.attention.key.bias', 'qformer.encoder.layer.7.attention.attention.key.bias', 'qformer.encoder.layer.9.output_query.dense.bias', 'qformer.encoder.layer.6.attention.attention.key.bias', 'qformer.encoder.layer.7.output_query.LayerNorm.weight', 'qformer.encoder.layer.5.intermediate_query.dense.bias', 'qformer.encoder.layer.4.attention.output.LayerNorm.weight', 'qformer.encoder.layer.9.intermediate_query.dense.weight', 'qformer.encoder.layer.0.attention.output.LayerNorm.weight', 'qformer.encoder.layer.2.crossattention.output.dense.bias', 'qformer.encoder.layer.10.attention.output.LayerNorm.bias', 'qformer.encoder.layer.10.output_query.LayerNorm.weight', 'qformer.encoder.layer.9.output_query.dense.weight', 'qformer.encoder.layer.0.attention.attention.value.bias', 'qformer.encoder.layer.4.attention.attention.value.bias', 'qformer.encoder.layer.1.attention.attention.query.bias', 'qformer.encoder.layer.2.output_query.LayerNorm.weight', 'qformer.encoder.layer.2.attention.attention.value.bias', 'qformer.encoder.layer.0.output_query.LayerNorm.bias', 'qformer.encoder.layer.3.output_query.LayerNorm.weight', 'qformer.encoder.layer.5.attention.output.dense.weight', 'qformer.encoder.layer.10.intermediate_query.dense.weight', 'qformer.encoder.layer.1.output_query.dense.weight', 'qformer.encoder.layer.0.intermediate_query.dense.weight', 'qformer.encoder.layer.1.attention.attention.query.weight', 'qformer.encoder.layer.5.output_query.dense.weight', 'qformer.encoder.layer.9.output_query.LayerNorm.bias', 'qformer.encoder.layer.10.crossattention.attention.value.weight', 'qformer.encoder.layer.10.crossattention.attention.query.bias', 'qformer.encoder.layer.1.attention.output.LayerNorm.weight', 'qformer.encoder.layer.0.crossattention.attention.query.bias', 'qformer.encoder.layer.0.attention.output.dense.bias', 'qformer.encoder.layer.6.crossattention.output.dense.bias', 'qformer.encoder.layer.2.output_query.LayerNorm.bias', 'qformer.encoder.layer.11.intermediate_query.dense.bias', 'qformer.encoder.layer.2.output_query.dense.bias', 'qformer.encoder.layer.2.attention.output.dense.bias', 'qformer.encoder.layer.6.crossattention.attention.key.bias', 'qformer.encoder.layer.5.output_query.LayerNorm.bias', 'qformer.encoder.layer.6.attention.output.LayerNorm.bias', 'qformer.encoder.layer.11.attention.attention.key.weight', 'qformer.encoder.layer.5.output_query.LayerNorm.weight', 'qformer.encoder.layer.4.crossattention.output.LayerNorm.bias', 'qformer.encoder.layer.1.output_query.LayerNorm.weight', 'qformer.encoder.layer.2.attention.attention.key.bias', 'qformer.encoder.layer.3.attention.attention.key.weight', 'qformer.encoder.layer.9.attention.attention.query.weight', 'qformer.encoder.layer.10.crossattention.output.LayerNorm.weight', 'qformer.encoder.layer.5.attention.output.LayerNorm.weight', 'qformer.encoder.layer.6.crossattention.output.dense.weight', 'qformer.encoder.layer.2.crossattention.attention.query.bias', 'qformer.encoder.layer.6.output_query.dense.weight', 'qformer.encoder.layer.8.crossattention.attention.value.weight', 'qformer.encoder.layer.1.attention.attention.key.weight', 'qformer.encoder.layer.9.attention.attention.query.bias', 'qformer.encoder.layer.10.attention.attention.value.bias', 'qformer.encoder.layer.10.output_query.dense.bias', 'qformer.encoder.layer.6.attention.attention.value.weight', 'qformer.encoder.layer.4.crossattention.attention.key.bias', 'qformer.encoder.layer.8.attention.attention.query.bias', 'qformer.encoder.layer.8.attention.attention.value.bias', 'qformer.encoder.layer.5.attention.attention.query.bias', 'qformer.encoder.layer.6.crossattention.attention.value.bias', 'qformer.encoder.layer.8.output_query.dense.weight', 'qformer.encoder.layer.4.output_query.LayerNorm.weight', 'qformer.encoder.layer.4.attention.output.dense.bias', 'qformer.encoder.layer.4.crossattention.attention.value.weight', 'qformer.encoder.layer.0.crossattention.output.LayerNorm.weight', 'qformer.encoder.layer.1.attention.attention.value.bias', 'qformer.encoder.layer.8.output_query.dense.bias', 'qformer.encoder.layer.8.crossattention.attention.key.weight', 'qformer.encoder.layer.8.output_query.LayerNorm.bias', 'qformer.encoder.layer.2.attention.output.LayerNorm.bias', 'qformer.encoder.layer.10.crossattention.output.dense.weight', 'qformer.encoder.layer.2.crossattention.output.LayerNorm.weight', 'qformer.encoder.layer.8.attention.attention.key.bias', 'qformer.encoder.layer.2.crossattention.output.LayerNorm.bias', 'qformer.encoder.layer.2.attention.attention.query.weight', 'qformer.encoder.layer.10.attention.attention.value.weight', 'qformer.encoder.layer.10.crossattention.attention.value.bias', 'qformer.encoder.layer.8.attention.attention.query.weight', 'qformer.encoder.layer.5.attention.attention.key.weight', 'qformer.encoder.layer.2.attention.attention.query.bias', 'qformer.encoder.layer.7.output_query.dense.bias', 'qformer.encoder.layer.10.crossattention.attention.key.bias', 'qformer.encoder.layer.9.attention.attention.key.weight', 'qformer.encoder.layer.10.attention.attention.query.bias', 'qformer.encoder.layer.2.output_query.dense.weight', 'qformer.encoder.layer.6.attention.attention.key.weight', 'qformer.encoder.layer.4.crossattention.output.dense.bias', 'qformer.encoder.layer.6.attention.attention.value.bias', 'qformer.encoder.layer.6.crossattention.output.LayerNorm.bias', 'qformer.encoder.layer.8.attention.attention.value.weight', 'qformer.encoder.layer.3.attention.attention.value.bias', 'qformer.encoder.layer.10.crossattention.attention.query.weight', 'qformer.encoder.layer.0.attention.attention.key.weight', 'qformer.encoder.layer.10.output_query.LayerNorm.bias', 'qformer.encoder.layer.2.crossattention.attention.value.weight', 'qformer.encoder.layer.10.attention.attention.key.bias', 'qformer.encoder.layer.0.crossattention.attention.key.bias', 'qformer.encoder.layer.9.attention.output.LayerNorm.bias', 'qformer.encoder.layer.1.attention.attention.key.bias', 'qformer.encoder.layer.4.crossattention.attention.key.weight', 'qformer.encoder.layer.11.intermediate_query.dense.weight', 'qformer.encoder.layer.8.crossattention.attention.query.weight', 'qformer.encoder.layer.8.crossattention.attention.key.bias', 'qformer.encoder.layer.6.attention.output.LayerNorm.weight', 'qformer.encoder.layer.8.intermediate_query.dense.bias', 'qformer.encoder.layer.3.output_query.LayerNorm.bias', 'qformer.encoder.layer.10.intermediate_query.dense.bias', 'qformer.encoder.layer.7.attention.output.dense.weight', 'qformer.encoder.layer.2.attention.attention.value.weight', 'qformer.encoder.layer.0.intermediate_query.dense.bias', 'qformer.encoder.layer.11.output_query.LayerNorm.bias', 'qformer.encoder.layer.8.attention.output.dense.weight', 'qformer.layernorm.weight', 'qformer.encoder.layer.5.attention.attention.query.weight', 'qformer.encoder.layer.8.attention.output.dense.bias', 'qformer.encoder.layer.0.output_query.dense.bias', 'qformer.encoder.layer.0.crossattention.attention.query.weight', 'qformer.encoder.layer.4.output_query.dense.bias', 'qformer.encoder.layer.9.intermediate_query.dense.bias', 'qformer.encoder.layer.2.intermediate_query.dense.weight', 'qformer.encoder.layer.1.intermediate_query.dense.bias', 'qformer.encoder.layer.11.attention.attention.value.bias', 'qformer.encoder.layer.9.attention.output.LayerNorm.weight', 'qformer.encoder.layer.4.attention.attention.key.weight', 'qformer.encoder.layer.7.intermediate_query.dense.bias', 'qformer.encoder.layer.4.intermediate_query.dense.bias', 'qformer.encoder.layer.6.crossattention.output.LayerNorm.weight', 'qformer.encoder.layer.8.attention.attention.key.weight', 'qformer.encoder.layer.6.crossattention.attention.value.weight', 'qformer.encoder.layer.1.output_query.dense.bias', 'qformer.encoder.layer.1.attention.output.LayerNorm.bias', 'qformer.encoder.layer.4.output_query.dense.weight', 'qformer.encoder.layer.8.crossattention.output.LayerNorm.weight', 'qformer.encoder.layer.3.attention.output.dense.bias', 'qformer.encoder.layer.10.crossattention.output.dense.bias', 'qformer.encoder.layer.7.attention.attention.key.weight', 'qformer.encoder.layer.11.attention.output.dense.bias', 'qformer.encoder.layer.2.intermediate_query.dense.bias', 'qformer.encoder.layer.2.crossattention.attention.query.weight', 'qformer.encoder.layer.4.attention.output.LayerNorm.bias', 'qformer.encoder.layer.5.attention.output.dense.bias', 'qformer.encoder.layer.9.output_query.LayerNorm.weight', 'qformer.encoder.layer.0.attention.attention.query.weight', 'qformer.encoder.layer.0.output_query.LayerNorm.weight', 'qformer.encoder.layer.4.crossattention.attention.query.weight', 'qformer.encoder.layer.0.crossattention.output.LayerNorm.bias', 'qformer.encoder.layer.7.attention.output.LayerNorm.weight', 'qformer.encoder.layer.5.attention.attention.value.weight', 'qformer.encoder.layer.6.output_query.LayerNorm.weight', 'qformer.encoder.layer.8.output_query.LayerNorm.weight', 'qformer.encoder.layer.4.attention.output.dense.weight', 'qformer.encoder.layer.0.attention.attention.value.weight', 
#               'qformer.encoder.layer.0.crossattention.attention.value.weight', 'qformer.encoder.layer.6.attention.attention.query.weight', 'qformer.encoder.layer.2.crossattention.attention.value.bias', 'qformer.encoder.layer.3.attention.attention.key.bias', 'qformer.encoder.layer.5.output_query.dense.bias', 'qformer.encoder.layer.3.attention.output.dense.weight', 'qformer.encoder.layer.4.crossattention.output.LayerNorm.weight', 'qformer.encoder.layer.10.attention.output.LayerNorm.weight', 'qformer.encoder.layer.4.crossattention.output.dense.weight', 'qformer.encoder.layer.10.attention.attention.query.weight', 'qformer.encoder.layer.6.attention.output.dense.weight', 'qformer.encoder.layer.6.crossattention.attention.query.bias', 'qformer.encoder.layer.10.attention.output.dense.weight', 'qformer.encoder.layer.10.crossattention.attention.key.weight', 'qformer.encoder.layer.11.attention.attention.key.bias', 'qformer.encoder.layer.3.attention.attention.query.bias', 'qformer.encoder.layer.3.attention.output.LayerNorm.weight', 'qformer.encoder.layer.9.attention.output.dense.bias', 'qformer.encoder.layer.0.output_query.dense.weight', 'qformer.encoder.layer.0.attention.attention.query.bias', 'qformer.encoder.layer.4.attention.attention.query.bias', 'qformer.encoder.layer.2.attention.output.LayerNorm.weight', 'qformer.encoder.layer.11.attention.attention.value.weight', 'qformer.encoder.layer.5.attention.attention.value.bias', 'qformer.encoder.layer.11.attention.output.dense.weight', 'qformer.encoder.layer.3.attention.output.LayerNorm.bias', 'qformer.encoder.layer.9.attention.attention.value.weight', 'qformer.encoder.layer.6.intermediate_query.dense.bias', 'qformer.encoder.layer.4.attention.attention.key.bias', 'qformer.encoder.layer.1.attention.output.dense.weight', 'qformer.encoder.layer.0.crossattention.attention.key.weight', 'qformer.encoder.layer.6.intermediate_query.dense.weight', 'qformer.encoder.layer.11.attention.attention.query.bias', 'qformer.encoder.layer.5.attention.output.LayerNorm.bias', 'qformer.layernorm.bias', 'qformer.encoder.layer.7.attention.attention.value.bias', 'qformer.encoder.layer.3.intermediate_query.dense.weight', 'qformer.encoder.layer.2.attention.attention.key.weight', 'qformer.encoder.layer.4.crossattention.attention.query.bias', 'qformer.encoder.layer.9.attention.attention.value.bias', 'qformer.encoder.layer.11.attention.output.LayerNorm.bias', 'qformer.encoder.layer.6.attention.output.dense.bias', 'qformer.encoder.layer.3.attention.attention.query.weight', 'qformer.encoder.layer.8.attention.output.LayerNorm.weight', 'qformer.encoder.layer.0.attention.attention.key.bias', 'qformer.encoder.layer.2.attention.output.dense.weight', 'qformer.encoder.layer.3.attention.attention.value.weight', 'qformer.encoder.layer.6.output_query.dense.bias', 'qformer.encoder.layer.8.intermediate_query.dense.weight', 'qformer.encoder.layer.10.crossattention.output.LayerNorm.bias', 'qformer.encoder.layer.11.output_query.LayerNorm.weight', 'qformer.encoder.layer.5.attention.attention.key.bias', 'qformer.encoder.layer.6.attention.attention.query.bias', 'qformer.encoder.layer.6.crossattention.attention.key.weight', 'qformer.encoder.layer.1.output_query.LayerNorm.bias', 'qformer.encoder.layer.2.crossattention.output.dense.weight', 'qformer.encoder.layer.0.crossattention.output.dense.weight', 'qformer.encoder.layer.0.attention.output.LayerNorm.bias', 'qformer.encoder.layer.8.crossattention.output.dense.bias', 'qformer.encoder.layer.6.output_query.LayerNorm.bias', 'qformer.encoder.layer.7.attention.attention.query.weight', 'qformer.encoder.layer.0.crossattention.attention.value.bias', 'qformer.encoder.layer.8.crossattention.output.dense.weight', 'qformer.encoder.layer.8.crossattention.attention.query.bias', 'qformer.encoder.layer.7.attention.attention.value.weight', 'qformer.encoder.layer.10.attention.attention.key.weight', 'qformer.encoder.layer.11.attention.attention.query.weight', 'qformer.encoder.layer.3.intermediate_query.dense.bias', 'qformer.encoder.layer.2.crossattention.attention.key.bias', 'qformer.encoder.layer.1.intermediate_query.dense.weight', 'qformer.encoder.layer.11.output_query.dense.weight', 'qformer.encoder.layer.7.attention.attention.query.bias', 'qformer.encoder.layer.4.attention.attention.query.weight', 'qformer.encoder.layer.7.attention.output.LayerNorm.bias', 'qformer.encoder.layer.5.intermediate_query.dense.weight', 'qformer.encoder.layer.9.attention.output.dense.weight', 'qformer.encoder.layer.10.attention.output.dense.bias', 'qformer.encoder.layer.11.attention.output.LayerNorm.weight', 'qformer.encoder.layer.10.output_query.dense.weight', 'qformer.encoder.layer.0.attention.output.dense.weight', 'qformer.encoder.layer.4.crossattention.attention.value.bias', 'qformer.encoder.layer.7.output_query.dense.weight', 'qformer.encoder.layer.11.output_query.dense.bias', 'qformer.encoder.layer.8.crossattention.attention.value.bias', 'qformer.encoder.layer.4.output_query.LayerNorm.bias']

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
        image_path = os.path.join(self.image_dir, img_name+'.jpg')
        image = Image.open(image_path).convert('RGB')
        
        encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, return_tensors="pt")
        
        # Remove batch dimension
        for k,v in encoding.items():
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
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16)
    model = model.to('cuda:0')

    # Freeze vision encoder and language model, unfreeze Q-Former
    for name, param in model.named_parameters():
        if "qformer" not in name:
            param.requires_grad = False

    # # Print non-frozen parameters
    # list_of_params = []
    # print("Trainable Parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         list_of_params.append(name)
    # # print(list_of_params[:5])
    # # list_of_params = [x for x in list_of_params if x not in non_t_params]
    # # print(len(list_of_params))
    lora_config = LoraConfig(
    r=1,  # LoRA rank
    lora_alpha=8,  # Scaling factor
    lora_dropout=0.1,  # Dropout to avoid overfitting
    # target_modules=("..."),  # Specify Q-former layers
    init_lora_weights="gaussian"  # Initialize LoRA weights with Gaussian distribution
     )

    # model = prepare_model_for_kbit_training(model)
    # model = get_peft_model(model, lora_config)


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

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)

    # Training parameters
    num_epochs = 10
    learning_rate = 5e-5
    warmup_steps = 100
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer and scheduler
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = SGD(model.parameters(), lr=learning_rate)

    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Training loop
    best_eval_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            batch = {k: v.to('cuda:0') for k, v in batch.items()}
            labels = batch['input_ids'].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=labels  # Let the model handle shifting the labels
            )
            loss = outputs.loss
            # print(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        
        # Evaluation
        model.eval()
        total_eval_loss = 0
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            batch = {k: v.to('cuda:0') for k, v in batch.items()}
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
