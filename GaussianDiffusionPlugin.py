#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[2]:


import math
import matplotlib.pyplot as plt

# Requires TensorFlow >=2.11 for the GroupNormalization layer.
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv3D
from tensorflow.keras.callbacks import *

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        self.network = network  # denoiser or noise predictor
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

    def train_step(self, data):
        # Unpack the data
        (images, image_input_past1, image_input_past2), y = data
        
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            # 3. Sample random noise to be added to the images in the batch
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
            print("noise.shape:", noise.shape)
            
            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)
            print("images_t.shape:", images_t.shape)
            
            # 5. Pass the diffused images and time steps to the network
            pred_noise = self.network([images_t, t, image_input_past1, image_input_past2], training=True)
            print("pred_noise.shape:", pred_noise.shape)
            
            # 6. Calculate the loss
            loss = self.loss(noise, pred_noise)

        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Update the weights of the network
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss": loss}

    
    def test_step(self, data):
        # Unpack the data
        (images, image_input_past1, image_input_past2), y = data

        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        # 3. Sample random noise to be added to the images in the batch
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        
        # 4. Diffuse the images with noise
        images_t = self.gdf_util.q_sample(images, t, noise)
        
        # 5. Pass the diffused images and time steps to the network
        pred_noise = self.network([images_t, t, image_input_past1, image_input_past2], training=False)
        
        # 6. Calculate the loss
        loss = self.loss(noise, pred_noise)

        # 7. Return loss values
        return {"loss": loss}

# ## Hyperparameters

# In[4]:
import PyPluMA
import PyIO
class GaussianDiffusionPlugin:
 def input(self, inputfile):
  self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
  import numpy as np
  batch_size = 256
  import tensorflow as tf
  num_epochs = 800         # Just for the sake of demonstration
  total_timesteps = 750   # 1000
  norm_groups = 8          # Number of groups used in GroupNormalization layer
  learning_rate = 1e-4

  img_size_H = 32
  img_size_W = 64
  img_channels = 5

  first_conv_channels = 64
  channel_multiplier = [1, 2, 4, 8]
  widths = [first_conv_channels * mult for mult in channel_multiplier]
  has_attention = [False, False, True, True]
  num_res_blocks = 2  # Number of residual blocks

  from layers.diffusion import GaussianDiffusion
  from tensorflow.keras.models import load_model

  pretrained_encoder = load_model(PyPluMA.prefix()+"/"+self.parameters["pretrained"], compile=False)
  #pretrained_encoder = load_model('saved_models/encoder_cnn_56deg_5var.h5', compile=False)
  pretrained_encoder.summary()
  first_five_layers = pretrained_encoder.layers[:5]
  for i, layer in enumerate(first_five_layers):
    print(f"Layer {i}: {layer}")
  input_layer = pretrained_encoder.input
  output_layer = first_five_layers[-1].output
  pretrained_encoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)
  pretrained_encoder.summary()
  for layer in pretrained_encoder.layers:
    layer.trainable = False
  pretrained_encoder._name = 'encoder'

  from layers.denoiser import build_unet_model_c2, build_unet_model_c2_no_cross_attn, build_unet_model_c2_no_encoder, build_unet_model_c2_no_cross_attn_encoder
  network = build_unet_model_c2(
    img_size_H=img_size_H,
    img_size_W=img_size_W,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    first_conv_channels=first_conv_channels,
    activation_fn=keras.activations.swish,
    encoder=pretrained_encoder,
)
  network = build_unet_model_c2(
    img_size_H=img_size_H,
    img_size_W=img_size_W,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    first_conv_channels=first_conv_channels,
    activation_fn=keras.activations.swish,
    encoder=pretrained_encoder,
)

  ema_network = build_unet_model_c2(
    img_size_H=img_size_H,
    img_size_W=img_size_W,
    img_channels=img_channels,
    widths=widths,
    has_attention=has_attention,
    num_res_blocks=num_res_blocks,
    norm_groups=norm_groups,
    first_conv_channels=first_conv_channels,
    activation_fn=keras.activations.swish,
    encoder=pretrained_encoder,
)
  ema_network.set_weights(network.get_weights())  # Initially the weights are the same
  gdf_util = GaussianDiffusion(timesteps=total_timesteps)
  model = DiffusionModel(
    network=network,
    ema_network=ema_network,
    gdf_util=gdf_util,
    timesteps=total_timesteps,
)

  from plugins.GaussianDiffusion.utils.normalization import batch_norm, batch_norm_reverse
  from plugins.GaussianDiffusion.utils.metrics import lat_weighted_rmse_one_var, lat_weighted_acc_one_var
  resolution_folder = '56degree'
  resolution = '5.625'  #1.40625, 2.8125, 5.625
  var_num = '5'

  test_data_tf = np.load(self.parameters["inputdir"]+"/concat_2017_2018_" + resolution + "_" + var_num + "var.npy")
  test_data_tf = test_data_tf.transpose((0, 2, 3, 1))
  test_data_tf_norm = batch_norm(test_data_tf, test_data_tf.shape, batch_size=1460)
  test_data_tf_norm_pred = test_data_tf_norm[2:]
  test_data_tf_norm_past1 = test_data_tf_norm[:-2]
  test_data_tf_norm_past2 = test_data_tf_norm[1:-1]
  print(test_data_tf_norm_pred.shape, test_data_tf_norm_past1.shape, test_data_tf_norm_past2.shape)

  from plugins.GaussianDiffusion.utils.normalization import batch_norm, batch_norm_reverse
  from plugins.GaussianDiffusion.utils.metrics import lat_weighted_rmse_one_var, lat_weighted_acc_one_var

  import tensorflow as tf
  import numpy as np

  def generate_images(model, original_samples, original_samples_past1, original_samples_past2):
    """
    @model: trained denoiser
    @original_samples: it just provides the shape, does not involve generation
    @original_samples_past: conditions from the past
    """
    num_images = original_samples.shape[0]
    img_size_H = original_samples.shape[1]
    img_size_W = original_samples.shape[2]
    img_channels = original_samples.shape[3]
    total_timesteps = model.timesteps  # Ensure this is defined in your model

    # 1. Randomly sample noise (starting point for reverse process)
    samples = tf.random.normal(shape=(num_images, img_size_H, img_size_W, img_channels), dtype=tf.float32)
    
    # 2. Sample from the model iteratively
    for t in reversed(range(0, total_timesteps)):
        tt = tf.cast(tf.fill([num_images], t), dtype=tf.int64)
        pred_noise = model.ema_network.predict([samples, tt, original_samples_past1, original_samples_past2],
                                               verbose=0, 
                                               batch_size=num_images
                                              )
        samples = model.gdf_util.p_sample(pred_noise, samples, tt, clip_denoised=True)
        
    # 3. Return generated samples and original samples
    return original_samples, samples
    # return original_samples.numpy(), samples.numpy()

  def predict_autoregressive(model, initial_inputs, prediction_horizon):
    
    predictions = []
    
    original_sample, sample_past1, sample_past2 = initial_inputs[0], initial_inputs[1], initial_inputs[2]  # t, t-2, t-1

    for _ in range(prediction_horizon):
        # Predict the next time step
        original_sample, generated_sample = generate_images(model, original_sample, sample_past1, sample_past2)
        
        print("original_sample.shape:", original_sample.shape, "generated_sample.shape:", generated_sample.shape)
        
        # Append the prediction to the list of predictions
        predictions.append(generated_sample)

        sample_past1 = sample_past2
        sample_past2 = generated_sample
        

    # Concatenate predictions along the time steps axis
    predictions = np.concatenate(predictions, axis=0)
    return predictions

  channels = ['geopotential_500', 'temperature_850', 
            '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']

  num_sample = 1#2918  # 2918 for entire test set
  num_channel = 5
  num_lead = 6

  rmse_matrix = np.zeros((num_sample, num_channel, num_lead))
  acc_matrix = np.zeros((num_sample, num_channel, num_lead))


  for z in range(num_sample):
    #print("sample #", z)
    original_samples = tf.convert_to_tensor(test_data_tf_norm_pred[z:z+num_lead], dtype=tf.float32)
    original_samples_past1 = tf.convert_to_tensor(test_data_tf_norm_past1[z:z+num_lead], dtype=tf.float32)
    original_samples_past2 = tf.convert_to_tensor(test_data_tf_norm_past2[z:z+num_lead], dtype=tf.float32)
    
    print(original_samples.shape, original_samples_past1.shape, original_samples_past2.shape)
    
    initial_inputs = [tf.convert_to_tensor(original_samples[0:1], dtype=tf.float32),
                  tf.convert_to_tensor(original_samples_past1[0:1], dtype=tf.float32), 
                  tf.convert_to_tensor(original_samples_past2[0:1], dtype=tf.float32)
                 ]

    future_predictions = predict_autoregressive(model, initial_inputs, prediction_horizon=num_lead)

    original_samples_unnormlalized = batch_norm_reverse(test_data_tf, test_data_tf.shape, 1459, original_samples)
    generated_samples_unnormlalized = batch_norm_reverse(test_data_tf, test_data_tf.shape, 1459, future_predictions)
    
    print(original_samples_unnormlalized.shape, generated_samples_unnormlalized.shape)
    
    
    for i in range(num_channel):
        print(f'{channels[i]}:')
        for j in range(num_lead):
            # print(f't{num_lead*(j+1)}: {lat_weighted_rmse_one_var(original_samples_unnormlalized[j:j+1], generated_samples_unnormlalized[j:j+1], var_idx=i, resolution=5.625):.5f}')
            rmse_matrix[z][i][j] = lat_weighted_rmse_one_var(original_samples_unnormlalized[j:j+1], generated_samples_unnormlalized[j:j+1], var_idx=i, resolution=5.625)
            acc_matrix[z][i][j] = lat_weighted_acc_one_var(original_samples_unnormlalized[j:j+1], generated_samples_unnormlalized[j:j+1], var_idx=i, resolution=5.625, clim=test_data_tf)
        print('\n')

    # print("rmse_matrix.shape:", rmse_matrix.shape)
    # print("acc_matrix.shape:", acc_matrix.shape)

  rmse_matrix_mean = np.mean(rmse_matrix, axis=0)
  acc_matrix_mean = np.mean(acc_matrix, axis=0)
  rmse_matrix_mean = np.mean(rmse_matrix, axis=0)
  rmse_matrix_mean = np.mean(rmse_matrix, axis=0)
  rmse_matrix_mean = np.mean(rmse_matrix, axis=0)
  rmse_matrix_mean = np.mean(rmse_matrix, axis=0)
  rmse_matrix_mean = np.mean(rmse_matrix, axis=0)

