from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, util
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything

import pickle
import wandb
from pytorch_lightning.loggers import WandbLogger
# Removed unused: "from yaml import parse"

# Import your data module and autoencoder (ensure that ae.py exists in your path)
from ae import MyDataModule, AutoEncoder

def str2bool(v):
    # Helper for boolean arguments
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', '1'):
       return True
    elif v.lower() in ('no', 'false', 'f', '0'):
       return False
    else:
       raise ValueError('Boolean value expected.')

def main(hparams):
    seed_everything(1)
    # Log in to wandb (make sure to provide a valid API key)
    wandb.login(key="YOUR_WANDB_API_KEY", anonymous="allow")
    # Capture the run object
    run = wandb.init(project=hparams.project_name)  
    wandb_logger = WandbLogger()

    # Load your SBERT encoder
    encoder = SentenceTransformer(hparams.sbert)
    with open(f'../{hparams.emb_data}', "rb") as fIn:
        stored_data = pickle.load(fIn)
        datasets = {
            'train1': stored_data['train1'],
            'train2': stored_data['train2'],
            'val1': stored_data['val1'],
            'val2': stored_data['val2'],
        }

    dm = MyDataModule(datasets, encoder, hparams.batch_size, denoising=hparams.denoising)

    # Create your autoencoder model; 768 is fixed (SBERT embedding size)
    model = AutoEncoder(768, hparams.hidden_dim, hparams.lr)

    # Use the run name from the wandb run
    filename = f'ae-quora-den-{run.name}'
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        dirpath='ae_saved/',
        monitor='val_loss',
        mode='min',
        filename=filename
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,  # change to 0 if no GPU available
        max_epochs=hparams.epochs,
        deterministic=True,
        callbacks=[checkpoint_callback, lr_monitor],
        auto_lr_find=True,
    )
    # Optionally tune the model if needed
    # trainer.tune(model, dm)
    trainer.fit(model, dm)
    print(checkpoint_callback.best_model_score)
    wandb.log({'best_loss': checkpoint_callback.best_model_score})
    run.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sbert', type=str, default='stsb-xlm-r-multilingual')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=768)
    # Use custom conversion for booleans instead of type=bool
    parser.add_argument('--denoising', type=str2bool, default=False)
    parser.add_argument('--emb_data', type=str, default='quora.pkl')
    parser.add_argument('--project_name', type=str, default='ae_ae')
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()

    main(args)
