import os
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

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

class ImageCaptioningModule(LightningModule):
    def __init__(self, model_name, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        self.processor = Blip2Processor.from_pretrained(model_name)

        # Freeze vision encoder and language model, unfreeze Q-Former
        for name, param in self.model.named_parameters():
            if "qformer" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        labels = batch['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        outputs = self.model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        outputs = self.model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                      start_factor=1.0, 
                                                      end_factor=0.1, 
                                                      total_iters=self.trainer.estimated_stepping_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

def main():
    # Hyperparameters
    batch_size = 16
    num_workers = 4
    max_epochs = 5

    # Data
    train_dataset = ImageCaptioningDataset(
        image_dir='/leonardo_scratch/fast/EUHPC_E03_068/.cache/r2r/',
        captions_file='./train_annotations.json',
        processor=Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    )
    val_dataset = ImageCaptioningDataset(
        image_dir='/leonardo_scratch/fast/EUHPC_E03_068/.cache/r2r/',
        captions_file='./val_annotations.json',
        processor=Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    # Model
    model = ImageCaptioningModule("Salesforce/blip2-flan-t5-xl")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Logger
    logger = TensorBoardLogger("lightning_logs", name="image_captioning")

    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=4,  # Use all 4 GPUs
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()