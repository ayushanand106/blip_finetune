import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import os
import json
from PIL import Image

# Set CUDA device
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
        image_path = os.path.join(self.image_dir, img_name+'.jpg')
        image = Image.open(image_path).convert('RGB')
        encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, return_tensors="pt")
        
        # Remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        
        # Add decoder_input_ids
        decoder_input_ids = self.processor.tokenizer.build_inputs_with_special_tokens(encoding['input_ids'].tolist())
        encoding['decoder_input_ids'] = torch.tensor(decoder_input_ids)
        
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
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", device_map="cuda:0", torch_dtype=torch.float16)

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
    print(eval_dataset[0].keys())
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)

    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            labels = batch.pop("labels").to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        eval_loss = 0
        # with torch.no_grad():
        #     for batch in tqdm(eval_dataloader, desc="Evaluating"):
        #         input_ids = batch.pop("input_ids").to(device)
        #         pixel_values = batch.pop("pixel_values").to(device)
        #         labels = batch.pop("labels").to(device)


        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)


                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                eval_loss += loss.item()

        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Evaluation Loss: {avg_eval_loss:.4f}")

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_blip2_flan_t5")
    processor.save_pretrained("./fine_tuned_blip2_flan_t5")

if __name__ == "__main__":
    main()