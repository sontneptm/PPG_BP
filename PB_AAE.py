import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from PB_util import gradient_penalty
from functools import partial
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models

# load data
whole_data = np.loadtxt("ppg_bp_filtered.csv",delimiter=',', dtype=np.float32)
bp_data = whole_data[:,:2]
ppg_data = whole_data[:,2:]
ppg_data = (ppg_data-ppg_data.min())/(ppg_data.max()-ppg_data.min())

# hyper params
LATENT_DIM = 2
SPLIT_RATE = 0.2
EPOCHS = 300
PPG_LENGTH = len(ppg_data[0])
AE_LR = 1.46e-3
GEN_LR = 5.0e-5
DSC_LR = 1.46e-5
VALID_STEP = 20

xt, xv, yt, yv = train_test_split(ppg_data, bp_data, test_size = SPLIT_RATE, random_state = 123)

class Encoder(object):
    def __init__(self):
        self.model()

    def model(self):
        model = tf.keras.Sequential(name='Encoder')

        # Layer 1
        model.add(layers.Dense(1024, activation=tf.nn.swish, input_shape=[PPG_LENGTH]))
        model.add(layers.Dense(512, activation=tf.nn.swish))
        model.add(layers.Dense(LATENT_DIM))

        return model
# end of Encoder class

class Decoder(object):
    def __init__(self):
        self.model()

    def model(self):
        model = tf.keras.Sequential(name='Decoder')

        # Layer 1
        model.add(layers.Dense(512, activation=tf.nn.swish, input_shape=[LATENT_DIM]))
        model.add(layers.Dense(1024, activation=tf.nn.swish))
        model.add(layers.Dense(PPG_LENGTH, activation='sigmoid'))

        return model
# end of Decoder class

class Discriminator(object):
    def __init__(self):
        pass

    def model(self):
        model = tf.keras.Sequential(name='Discriminator')

        # Layer 1
        model.add(layers.Dense(1024, activation=tf.nn.swish, input_shape=[LATENT_DIM]))
        model.add(layers.Dense(1024, activation=tf.nn.swish))

        return model
# end of Discriminator class

def train():
    # build models
    enc = Encoder().model()
    dec = Decoder().model()
    dsc = Discriminator().model()

    # set optimizers
    opt_ae = tf.keras.optimizers.Adam(learning_rate=AE_LR, beta_1=0.5, beta_2=0.999, epsilon=0.01)
    opt_gen = tf.keras.optimizers.Adam(learning_rate=GEN_LR, beta_1=0.5, beta_2=0.999, epsilon=0.01)
    opt_dsc = tf.keras.optimizers.Adam(learning_rate=DSC_LR, beta_1=0.5, beta_2=0.999, epsilon=0.01)

    # Set trainable variables
    var_ae = enc.trainable_variables + dec.trainable_variables
    var_gen = enc.trainable_variables
    var_dsc = dsc.trainable_variables

    # Check point
    check_point_dir = os.path.join(os.getcwd(), 'training_checkpoints')

    graph_path = os.path.join(os.getcwd(), 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(os.getcwd(), 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    check_point_prefix = os.path.join(check_point_dir, 'aae')

    enc_name = 'aae_enc'
    dec_name = 'aae_dec'
    dsc_name = 'aae_dsc'

    graph = 'aae'

    check_point = tf.train.Checkpoint(opt_gen=opt_gen, opt_dcs=opt_dsc, opt_ae=opt_ae, encoder=enc, decoder=dec, discriminator=dsc)
    ckpt_manager = tf.train.CheckpointManager(check_point, check_point_dir, max_to_keep=5,checkpoint_name=check_point_prefix)

    # define generator training step
    def gen_training_step(x):
        with tf.GradientTape() as gen_tape:
            z_gen = enc(x, training = True)
            #z_gen_input = tf.concat([y, z_gen], axis=-1, name='z_gen_input')
            z_gen_input= z_gen

            dsc_fake = dsc(z_input, training=True)
            loss_gen = -tf.reduce_mean(dsc_fake)
        grad_gen = gen_tape.gradient(loss_gen, var_gen)
        opt_gen.apply_gradients(zip(grad_gen, var_gen))

    #define AE training step
    def training_step(x):
        with tf.GradientTape() as ae_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as dsc_tape:
            z = bp_data # real data
            z_gen = enc(np.asmatrix(x), training=True)

            #z_input = tf.concat([y,z], axis=-1, name='z_input')
            z_input = z
            #z_gen_input = tf.concat([y,z_gen], axis=-1, name='z_input')
            z_gen_input = z_gen

            x_bar = dec(z_gen_input, training=True)

            dsc_real = dsc(z_input, training=True)
            dsc_fake = dsc(z_gen_input, training=True)

            real_loss, fake_loss = -tf.reduce_mean(dsc_real), tf.reduce_mean(dsc_fake)
            gp = gradient_penalty(partial(dsc, training=True), z_input, z_gen_input)

            loss_gen = -tf.reduce_mean(dsc_fake)
            loss_dsk = (real_loss + fake_loss) + gp * 0.1
            loss_ae = tf.reduce_mean(tf.abs(tf.subtract(x, x_bar)))
        
        grad_ae = ae_tape.gradient(loss_ae, var_ae)
        grad_gen = gen_tape.gradient(loss_gen, var_gen)
        grad_dsc = dsc_tape.gradient(loss_dsc, var_dsc)

        opt_ae.apply_gradients(zip(grad_ae, var_ae))
        opt_gen.apply_gradients(zip(grad_gen, var_gen))
        opt_dsc.apply_gradients(zip(grad_dsc, var_dsc))

    #define validation step
    def validation_step(x):
        z = bp_data
        z_gen = enc(x, traning=False)
        z_input = z
        z_gen_input = z_gen

        x_bar = dec(z_gen_input, training=False)
        x_gen = dec(z_input, training=False)

        dsc_real = dsc(z_input, training=False)
        dsc_fake = dsc(z_gen_input, training=False)

        real_loss, fake_loss = -tf.reduce_mean(dsc_real), tf.reduce_mean(dsc_fake)
        gp = gradient_penalty(partial(dsc, training=False), z_input, z_gen_input)

        loss_gen = -tf.reduce_mean(dsc_fake)
        loss_dsc = (real_loss + fake_loss) + gp * 0.1
        loss_ae = tf.reduce_mean(tf.abs(tf.subtract(x, x_bar)))

        return x_gen, x_bar, loss_dsc.numpy(), loss_gen.numpy(), loss_ae.numpy(), (fake_loss.numpy()-real_loss.numpy())

    # do TRAIN

    start_time = time.time()
    for epoch in range(EPOCHS):
        # train AAE
        num_train = 0
        for d in xt:
            if num_train % 2 == 0:
                training_step(d)
            else :
                gen_training_step(d)
                gen_training_step(d)
            num_train +=1
        
        # validation
        num_valid = 0
        val_loss_dsc, val_loss_gen, val_loss_ae, val_was_x = [],[],[],[]
        for d in xv:
            x_gen, x_bar, loss_dsc, loss_gen, loss_ae, was_x = validation_step(d)

            val_loss_dsc.append(loss_dsc)
            val_loss_gen.append(loss_gen)
            val_loss_ae.append(loss_ae)
            val_was_x.append(was_x)

            num_valid += 1

            if num_valid > VALID_STEP:
                break
        
        elapsed_time = (time.time() - start_time) /60.
        val_loss_ae = np.mean(np.reshape(val_loss_ae, (-1)))
        val_loss_dsc = np.mean(np.reshape(val_loss_dsc, (-1)))
        val_loss_gen = np.mean(np.reshape(val_loss_gen, (-1)))
        val_was_x = np.mean(np.reshape(val_was_x, (-1)))

        print("[Epoch: {:04d}] {:.01f}m.\tdis: {:.6f}\tgen: {:.6f}\tae: {:.6f}\tw_x: {:.6f}".format(epoch, elapsed_time, val_loss_dis, val_loss_gen, val_loss_ae, val_was_x))
        
        if epoch % param.save_frequency == 0 and epoch > 1:
            save_decode_image_array(x_valid.numpy(), path=os.path.join(graph_path,'{}_original-{:04d}.png'.format(graph, epoch)))
            save_decode_image_array(x_bar.numpy(), path=os.path.join(graph_path, '{}_decode-{:04d}.png'.format(graph, epoch)))
            save_decode_image_array(x_tilde.numpy(), path=os.path.join(graph_path, '{}_generated-{:04d}.png'.format(graph, epoch)))
            ckpt_manager.save(checkpoint_number=epoch)

    save_message = "\tSave model: End of training"

    enc.save_weights(os.path.join(model_path, enc_name))
    dec.save_weights(os.path.join(model_path, dec_name))
    dsc.save_weights(os.path.join(model_path, dsc_name))

    # 6-3. Report
    print("[Epoch: {:04d}] {:.01f} min.".format(EPOCHS, elapsed_time))
    print(save_message)
    """
    x_axis = range(len(ppg_data[0]))
    encoded_val_data = enc(ppg_data).numpy()
    decoded_val_data = dec(encoded_val_data).numpy()
    plt.plot(x_axis, decoded_val_data[5])
    plt.plot(x_axis, ppg_data[5])
    plt.show()
    """
    pd = enc.predict(ppg_data)
   
    for i in range(len(ppg_data)):
        #fmt = '실제값: {1}, 예측값: {2:.5f} {3:.5f}, 정제된값: {4:.0f} {5:.0f}'
        #print(fmt.format(yv[i], pd[i][0], pd[i][1], pd[i][0], pd[i][1]))
        print("실제값 : ", bp_data[i],"\t", "예측값 :", pd[i])
    
    print(r2_score(bp_data, pd))

# end of train() method

if __name__ == "__main__":
    train()
    