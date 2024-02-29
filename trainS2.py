# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf

import dataloader as DL
import layers as CLayers
import train_utils as TU
import models as MD

from sklearn.model_selection import train_test_split
# -

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# ## Load data

df = pd.read_csv('./data/ADNI_saliency.csv', low_memory=False)
date = '20240128_1800'
input_name = 'AbsDiff'
unique_rids = df['RID'].unique()

# +
# train_rids, valid_rids = train_test_split(unique_rids, test_size=0.2, random_state=2024)

# +
# with open(f'./data/train_rids_S2.txt', 'w') as file:
#     for rid in train_rids:
#         file.write(f"{rid}\n")
# with open(f'./data/val_rids_S2.txt', 'w') as file:
#     for rid in valid_rids:
#         file.write(f"{rid}\n")
# -

with open(f'./data/train_rids_S2.txt', 'r') as file:
    train_rids = [line.strip() for line in file]
train_rids = [int(float(rid)) for rid in train_rids]
# Load validation RIDs from the text file
with open(f'./data/val_rids_S2.txt', 'r') as file:
    valid_rids = [line.strip() for line in file]
valid_rids = [int(float(rid)) for rid in valid_rids]

ds_config = dict(
    source_dir="/ngochuynh/f/Dataset/ADNI",
    filepath="./data/ADNI_saliency.csv",
    input_name="AbsDiff",
    batch_size = 4,
    timesteps = 12,
    img_shape = (189,216,189),
    sal_shape = (84,48,42),
    pe_dim    = 128,
)

train_ds = DL.InputFunctionS2(**ds_config, list_rids=train_rids, shuffle=True)
valid_ds = DL.InputFunctionS2(**ds_config, list_rids=valid_rids, shuffle=False)

# +
# check_nan = dict(
#     step=[],
#     mri_nan=[],
#     pe_nan=[],
#     sal_nan=[]
# )
# for i, (feat, label) in enumerate(train_ds()):
#     print(f'Step {i}')
#     if np.isnan(feat['mri_image'].numpy()).any():
#         print('MRI image contains NaN')
#         check_nan['step'].append(i)
#         check_nan['mri_nan'].append(1)
#     else:
#         print('MRI image does NOT contain NaN')
    
#     if np.isnan(feat['pos_enc'].numpy()).any():
#         print('PE contains NaN')
#         check_nan['step'].append(i)
#         check_nan['pe_nan'].append(1)
#     else:
#         print('PE does NOT contain NaN')
    
#     if np.isnan(label['sal_image'].numpy()).any():
#         print('Saliency map contains NaN')
#         check_nan['step'].append(i)
#         check_nan['sal_nan'].append(1)
#     else:
#         print('Saliency map does NOT contain NaN')

# +
# check_nan
# -



# ## Create model

model_config = dict(
    enc_filter=16,
    gen_filter=16,
    dec_filter=16,
    dic_filter=16,
    enc_dropout=0.3,
    dic_dropout=0.3,
    latent_dim=1024,
)

model = MD.ProGAN(input_shape=(189,216,189,1), latent_shape=(1024+128,), dicrim_shape=(84,48,42,1))
model._make_model(**model_config)

# ## Create trainer

num_epochs = 50
optimizers = dict(
    dec_op  = tf.keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-6),
    cons_op = tf.keras.optimizers.Adam(learning_rate=1e-4, weight_decay=1e-6),
    gen_op  = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-6, beta_1=0.5, beta_2=0.999),
    dis_op  = tf.keras.optimizers.Adam(learning_rate=1e-5, weight_decay=1e-6, beta_1=0.5, beta_2=0.999),
)

trainer = TU.TrainAndEvaluateS2(
    model=model,
    model_dir=f"./checkpoints/S2_unimodel_{input_name}_{date}",
    input_name=input_name,
    train_dataset=train_ds(),
    eval_dataset=valid_ds(),
    num_epochs=num_epochs,
    timepoints=12,
    optimizers=optimizers,
    pretrained='epoch_17',
)

trainer.train_and_evaluate()


