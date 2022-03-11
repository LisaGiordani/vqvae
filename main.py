
import data, cudafy, scheduler, vq, encoders, decoders, vqvae, vqvae_training, transformer, transformer_training

import torch
import os

dataset="dog" # "dog" or "CIFAR"
generation_model="transformer" # "transformer" or "pixelCNN"

VAE_PATH='vae.pt'

# Data
imagesIntorch = data.preprocess(dataset)
nb_images = imagesIntorch.shape[0]

# Model VQ-VAE-2
model = vqvae.make_vae(dataset)
if os.path.exists(VAE_PATH):
    model.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
model.to(device)

# Training of VQ-VAE-2
epochs = 5000 # init: 5000
batch_size = 256 #16
lr = 0.001 # init : 0.001
vqvae_training.train_vae(model, imagesIntorch, epochs, batch_size, lr)

# Image reconstruction
Image.open('reconstructions.png')

# Training of the generation model
if generation_model=="transformer":
    hparams = {'resume_from_checkpoint': None,
        'folder': '/out/generator"', 
        'epochs': 5, #200 
        'lr': 1e-3, 
        'weight_decay': 0, 
        'scheduler_gamma':  1, 
        'batch_size': 128, # ok : 32, 64, 128 / not ok : 256
        'num_workers': 1, 
        'nb_examples':2000, #nb_images
        'vqvae_model_path':VAE_PATH, 
        'gpus': 1 if torch.cuda.is_available() else 0}
    generator, trainergenerator = transformer_training.train_transformer(hparams)

# Image generation
if generation_model=="transformer":
    transformer_generation.random_generation(generator, nb_examples=2)