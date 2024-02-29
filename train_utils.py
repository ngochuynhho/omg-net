import time
import numpy as np
import tensorflow as tf
import tensorflow.compat.v2.summary as summary

from scipy import ndimage
from tensorflow.python.ops import summary_ops_v2

import losses as LO

class TrainAndEvaluateS1:
    def __init__(self,
                 model,
                 model_dir,
                 train_dataset,
                 eval_dataset,
                 num_epochs,
                 train_steps,
                 optimizer):
        self.num_epochs  = num_epochs
        self.train_steps = train_steps
        self.model_dir   = model_dir
        self.model       = model
        
        self.train_ds = train_dataset
        self.val_ds   = eval_dataset
        
        self.optimizer     = optimizer
        self.loss_function = LO.custom_focal_loss_v2
        
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric   = tf.keras.metrics.Mean(name="val_loss")
        self.val_metrics   = LO.CustomMetricS1()
        
        self.save_ckpt     = False
        self.best_val_loss = 1e6
    
    @tf.function(reduce_retracing=True)
    def train_one_step(self, x, y):
        with tf.GradientTape() as tape:
            labels = y['conv_lb']
            logits = self.model(x, training=True)
            train_loss = self.loss_function(labels, logits)

        with tf.name_scope("gradients"):
            grads = tape.gradient(train_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return train_loss
    
    def train_and_evaluate(self):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=self.optimizer,
            model=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, str(self.model_dir), max_to_keep=None)
        
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f"Latest checkpoint restored from {ckpt_manager.latest_checkpoint}.")
            
        train_summary_writer = summary.create_file_writer(
            self.model_dir + "/train")
        val_summary_writer = summary.create_file_writer(
            self.model_dir + "/valid")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            with train_summary_writer.as_default():
                self.train_one_epoch(ckpt.step, epoch)

            # Run a validation loop at the end of each epoch.
            with val_summary_writer.as_default():
                self.evaluate(ckpt.step)
            
            if self.save_ckpt:
                save_path = ckpt_manager.save()
                print(f"Saved checkpoint for step {ckpt.step.numpy()}: {save_path}")
        
    def train_one_epoch(self, step_counter, epoch):
        starttime = time.time()
        for i, (x, y) in enumerate(self.train_ds):
            step = int(step_counter)
            train_loss = self.train_one_step(x, y)
            if step == 0:
                func = self.train_one_step.get_concrete_function(x, y)
                summary_ops_v2.graph(func.graph)

            # Update training metric.
            self.train_loss_metric.update_state(train_loss)

            # Log every 200 batches.
            if step % 2 == 0:
                # Display metrics
                mean_loss = self.train_loss_metric.result()
                #print(f"epoch {self.epoch} - step {step}: mean loss = {mean_loss:.4f} -- {time.time()-starttime:.3f} sec")
                # save summaries
                summary.scalar("loss", mean_loss, step=step_counter)
                # Reset training metrics
                self.train_loss_metric.reset_states()
            
            step_counter.assign_add(1)
        print(f"Traing time: {time.time()-starttime:.3f} sec")
    
    @tf.function(reduce_retracing=True)
    def evaluate_one_step(self, x, y):
        labels = y['conv_lb']
        logits = self.model(x, training=False)
        valid_loss = self.loss_function(labels, logits)
        return valid_loss, labels, logits
    
    def evaluate(self, step_counter):
        starttime = time.time()
        
        self.val_metrics.reset_states()
        
        for x_val, y_val in self.val_ds:
            val_loss, val_labels, val_logits = self.evaluate_one_step(x_val, y_val)

            # Update val metrics
            self.val_loss_metric.update_state(val_loss)
            self.val_metrics.update_state(y_val, val_logits)

        val_loss = self.val_loss_metric.result()
        if tf.less(val_loss, self.best_val_loss):
            self.save_ckpt = True
            self.best_val_loss = val_loss
        else:
            self.save_ckpt = True
        summary.scalar("loss",
                       val_loss,
                       step=step_counter)
        self.val_loss_metric.reset_states()
        
        val_results = self.val_metrics.result()
        for key, value in val_results.items():
             summary.scalar(key, value, step=step_counter)

        print(f"Validation - epoch {self.epoch}: loss = {val_loss:.3f} -- acc = {val_results['accuracy']:.3f} -- f1 = {val_results['f1']:.3f} -- {time.time()-starttime:.3f} sec")
    
    def save_model(self, save_name):
        self.model.save_weights(self.model_dir+'/'+save_name)
    
    def load_model(self, save_name):
        self.model.load_weights(self.model_dir+'/'+save_name)

class TrainAndEvaluateS1_WSL:
    def __init__(self,
                 model,
                 model_dir,
                 input_name,
                 train_dataset,
                 eval_dataset,
                 num_epochs,
                 train_steps,
                 optimizer):
        self.num_epochs  = num_epochs
        self.train_steps = train_steps
        self.model_dir   = model_dir
        self.model       = model
        self.input_name  = input_name
        
        self.train_ds = train_dataset
        self.val_ds   = eval_dataset
        
        self.optimizer     = optimizer
        self.loss_function = LO.custom_focal_loss_v2
        
        self.train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss_metric   = tf.keras.metrics.Mean(name="val_loss")
        self.val_metrics   = LO.CustomMetricS1()
        
        self.save_ckpt     = False
        self.best_val_loss = 1e6
    
    @tf.function(reduce_retracing=True)
    def train_one_step(self, x, y):
        with tf.GradientTape() as tape:
            inputs = x[self.input_name]
            neg_x1 = x[f"{self.input_name}_neg_1"]
            neg_x2 = x[f"{self.input_name}_neg_2"]
            labels = y['conv_lb']
            neg_lb_1 = y['conv_lb_neg1']
            neg_lb_2 = y['conv_lb_neg2']
            label_concat = tf.concat((labels, neg_lb_1, neg_lb_2), axis=0)

            _, att_map1    = self.model(neg_x1, training=False)
            resized_neg_x1 = tf.numpy_function(random_select_and_resize, [neg_x1, att_map1], tf.float32)
            norm_neg_x1    = normalize_minmax(resized_neg_x1)
            aug_neg_x1     = tf.math.add(0.75*neg_x1, 0.25*tf.math.multiply(neg_x1, norm_neg_x1))
            _, att_map2    = self.model(neg_x2, training=False)
            resized_neg_x2 = tf.numpy_function(random_select_and_resize, [neg_x2, att_map2], tf.float32)
            norm_neg_x2    = normalize_minmax(resized_neg_x2)
            aug_neg_x2     = tf.math.add(0.75*neg_x2, 0.25*tf.math.multiply(neg_x2, norm_neg_x2))
            
            logits, _ = self.model(tf.concat((inputs, aug_neg_x1, aug_neg_x2), axis=0), training=True)
            train_loss = self.loss_function(label_concat, logits, alpha=0.65, gamma=1)

        with tf.name_scope("gradients"):
            grads = tape.gradient(train_loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return train_loss
    
    def train_and_evaluate(self):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=self.optimizer,
            model=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, str(self.model_dir), max_to_keep=None)
        
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f"Latest checkpoint restored from {ckpt_manager.latest_checkpoint}.")
            
        train_summary_writer = summary.create_file_writer(
            self.model_dir + "/train")
        val_summary_writer = summary.create_file_writer(
            self.model_dir + "/valid")

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            with train_summary_writer.as_default():
                self.train_one_epoch(ckpt.step, epoch)

            # Run a validation loop at the end of each epoch.
            with val_summary_writer.as_default():
                self.evaluate(ckpt.step)
            
            if self.save_ckpt:
                save_path = ckpt_manager.save()
                print(f"Saved checkpoint for step {ckpt.step.numpy()}: {save_path}")
        
    def train_one_epoch(self, step_counter, epoch):
        starttime = time.time()
        for i, (x, y) in enumerate(self.train_ds):
            step = int(step_counter)
            train_loss = self.train_one_step(x, y)
            if step == 0:
                func = self.train_one_step.get_concrete_function(x, y)
                summary_ops_v2.graph(func.graph)

            # Update training metric.
            self.train_loss_metric.update_state(train_loss)

            # Log every 200 batches.
            if step % 2 == 0:
                # Display metrics
                mean_loss = self.train_loss_metric.result()
                #print(f"epoch {self.epoch} - step {step}: mean loss = {mean_loss:.4f} -- {time.time()-starttime:.3f} sec")
                # save summaries
                summary.scalar("loss", mean_loss, step=step_counter)
                # Reset training metrics
                self.train_loss_metric.reset_states()
            
            step_counter.assign_add(1)
        print(f"Traing time: {time.time()-starttime:.3f} sec")
    
    @tf.function(reduce_retracing=True)
    def evaluate_one_step(self, x, y):
        inputs = x[self.input_name]
        labels = y['conv_lb']
        logits, _ = self.model(inputs, training=False)
        valid_loss = self.loss_function(labels, logits, alpha=0.95, gamma=2)
        return valid_loss, labels, logits
    
    def evaluate(self, step_counter):
        starttime = time.time()
        
        self.val_metrics.reset_states()
        
        for x_val, y_val in self.val_ds:
            val_loss, val_labels, val_logits = self.evaluate_one_step(x_val, y_val)

            # Update val metrics
            self.val_loss_metric.update_state(val_loss)
            self.val_metrics.update_state(y_val, val_logits)

        val_loss = self.val_loss_metric.result()
        if tf.less(val_loss, self.best_val_loss):
            self.save_ckpt = True
            self.best_val_loss = val_loss
        else:
            self.save_ckpt = True
        summary.scalar("loss",
                       val_loss,
                       step=step_counter)
        self.val_loss_metric.reset_states()
        
        val_results = self.val_metrics.result()
        for key, value in val_results.items():
             summary.scalar(key, value, step=step_counter)

        print(f"Validation - epoch {self.epoch}: loss = {val_loss:.3f} -- acc = {val_results['accuracy']:.3f} -- f1 = {val_results['f1']:.3f} -- {time.time()-starttime:.3f} sec")
    
    def save_model(self, save_name):
        self.model.save_weights(self.model_dir+'/'+save_name)
    
    def load_model(self, save_name):
        self.model.load_weights(self.model_dir+'/'+save_name)
        
class TrainAndEvaluateS2:
    def __init__(self,
                 model,
                 model_dir,
                 input_name,
                 train_dataset,
                 eval_dataset,
                 num_epochs,
                 timepoints,
                 latent_dim,
                 pe_dim,
                 optimizers,
                 pretrained=None):
        self.num_epochs = num_epochs
        self.timepoints = timepoints
        self.latent_dim = latent_dim
        self.pe_dim     = pe_dim
        self.model_dir  = model_dir
        self.model      = model
        self.input_name = input_name
        self.pretrained = pretrained
        
        self.train_ds = train_dataset
        self.val_ds   = eval_dataset
        
        self.optimizers    = optimizers
        self.decode_loss_func = LO.kld_loss
        self.CL_func = LO.ContrastiveLoss(self.latent_dim, 0.1, 200)
        self.gen_loss_func = LO.bce_one_var_loss
        self.dis_loss_func = LO.bce_two_var_loss
        self.mse_loss_func = LO.meam_squred_error_loss
        
        self.train_loss_metric = dict(
            dec_loss=tf.keras.metrics.Mean(name="train_dec_loss"),
            cons_loss=tf.keras.metrics.Mean(name="train_cons_loss"),
            gen_loss=tf.keras.metrics.Mean(name="train_gen_loss"),
            dis_loss=tf.keras.metrics.Mean(name="train_dis_loss"),
        )
        self.val_loss_metric   = dict(
            dec_loss=tf.keras.metrics.Mean(name="valid_dec_loss"),
            cons_loss=tf.keras.metrics.Mean(name="valid_cons_loss"),
            gen_loss=tf.keras.metrics.Mean(name="valid_gen_loss"),
            dis_loss=tf.keras.metrics.Mean(name="valid_dis_loss"),
        )
        #self.val_metrics   = LO.CustomMetricS2()
        
        self.save_ckpt     = False
        self.best_val_loss = 1e6
        
        if self.pretrained:
            self.load_model(pretrained)
            self.start_epoch = int(pretrained.split('_')[-1]) + 1
        else:
            self.start_epoch = 0
        
    @tf.function(reduce_retracing=True)
    def train_one_step(self, feat, label):
        x  = feat["mri_image"]
        pe = feat["pos_enc"]
        y  = label["sal_image"]
        m  = label["mask_visits"]
        flat_m = tf.cast(tf.reshape(m, shape=[-1]), dtype=tf.int32)
        
        with tf.GradientTape() as e1_tape, tf.GradientTape() as e2_tape, tf.GradientTape() as g_tape, tf.GradientTape() as di_tape:
            feat_e1 = self.model.E1(x, training=True)
            mean_e1, logvar_e1 = self.model.encode(feat_e1)
            z_e1 = self.model.reparameterize(mean_e1, logvar_e1)
            latent_e1 = tf.concat((z_e1, pe[:,0,:]), axis=-1)
            
            out_de = self.model.De(latent_e1, training=True)
            #dec_loss = self.decode_loss_func(x, out_de, z_e1, mean_e1, logvar_e1, mode='mean')
            dec_loss = self.decode_loss_func(x, out_de, mean_e1, logvar_e1)
            
            out_gens  = []
            out_reals = []
            cons_loss = 0.
            for i in range(self.timepoints):   
                if i==0:
                    out_gen = self.model.G1(latent_e1, training=True)
                else:
                    feat_y = y[:,i-1,:,:,:,:]
                    feat_e2 = self.model.E2(feat_y, training=True)
                    mean_e2, logvar_e2 = self.model.encode(feat_e2)
                    z_e2 = self.model.reparameterize(mean_e2, logvar_e2)
                    latent_e2 = tf.concat((z_e1+z_e2, pe[:,i,:]), axis=-1)
                    out_gen = self.model.G1(latent_e2, training=True)
                    
                    cons_loss += self.CL_func.contrastive_loss(z_e1, z_e2)
                    
                    out_reals.append(feat_y)
                out_gens.append(out_gen)
            
            out_reals.append( y[:,-1,:,:,:,:])
            out_reals = tf.concat(out_reals, axis=0)
            out_gens  = tf.concat(out_gens, axis=0)
            
            out_reals = tf.boolean_mask(out_reals, flat_m, axis=0)
            out_gens  = tf.boolean_mask(out_gens, flat_m, axis=0)
            
            real_logits = self.model.Di(out_reals, training=True)
            fake_logits = self.model.Di(out_gens, training=True)
            
            gen_loss = self.gen_loss_func(fake_logits) + self.mse_loss_func(out_reals, out_gens)
            dis_loss = self.dis_loss_func(real_logits, fake_logits)
        
        with tf.name_scope("gradients"):
            dec_gradient = e1_tape.gradient(dec_loss, self.model.E1.trainable_variables+self.model.De.trainable_variables)
            cons_gradient = e2_tape.gradient(cons_loss, self.model.E1.trainable_variables+self.model.E2.trainable_variables)
            gen_gradient = g_tape.gradient(gen_loss, self.model.E1.trainable_variables+self.model.E2.trainable_variablesself.model.G1.trainable_variables)
            dis_gradient = di_tape.gradient(dis_loss, self.model.Di.trainable_variables)
            
            self.optimizers["dec_op"].apply_gradients(zip(dec_gradient, self.model.E1.trainable_variables+self.model.De.trainable_variables))
            self.optimizers["cons_op"].apply_gradients(zip(cons_gradient, self.model.E1.trainable_variables+self.model.E2.trainable_variables))
            self.optimizers["gen_op"].apply_gradients(zip(gen_gradient, self.model.E1.trainable_variables+self.model.E2.trainable_variables+self.model.G1.trainable_variables))
            self.optimizers["dis_op"].apply_gradients(zip(dis_gradient, self.model.Di.trainable_variables))
        
        return {
            'dec_loss':dec_loss,
            'cons_loss':cons_loss,
            'gen_loss':gen_loss,
            'dis_loss':dis_loss,
        }
    
    def train_one_epoch(self, epoch):
        starttime = time.time()
        print()
        for i, (x, y) in enumerate(self.train_ds):
            train_loss = self.train_one_step(x, y)

            # Update training metric.
            #self.train_loss_metric.update_state(sum(train_loss.values()))
            self.train_loss_metric['dec_loss'].update_state(train_loss['dec_loss'])
            self.train_loss_metric['cons_loss'].update_state(train_loss['cons_loss'])
            self.train_loss_metric['gen_loss'].update_state(train_loss['gen_loss'])
            self.train_loss_metric['dis_loss'].update_state(train_loss['dis_loss'])

            # Log every 200 batches.
            if i % 2 == 0:
                print('\r', end='', flush=True)
                # Display metrics
                mean_dec_loss = self.train_loss_metric['dec_loss'].result()
                mean_cons_loss = self.train_loss_metric['cons_loss'].result()
                mean_gen_loss = self.train_loss_metric['gen_loss'].result()
                mean_dis_loss = self.train_loss_metric['dis_loss'].result()
                # Reset training metrics
                self.train_loss_metric['dec_loss'].reset_states()
                self.train_loss_metric['cons_loss'].reset_states()
                self.train_loss_metric['gen_loss'].reset_states()
                self.train_loss_metric['dis_loss'].reset_states()
                print(f'step {i} : dec_loss = {mean_dec_loss:.3f} -- cons_loss = {mean_cons_loss:.3f} -- gen_loss = {mean_gen_loss:.3f} -- dis_loss = {mean_dis_loss:.3f}', end='', flush=True)

        print(f' -- Traing time: {time.time()-starttime:.3f} sec')
    
    def train_and_evaluate(self):            
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs, 1):
            #self.epoch = epoch
            self.train_one_epoch(epoch)

            # Run a validation loop at the end of each epoch.
            self.evaluate(epoch)
    
    @tf.function(reduce_retracing=True)
    def evaluate_one_step(self, feat, label):
        x  = feat["mri_image"]
        pe = feat["pos_enc"]
        y  = label["sal_image"]
        m  = label["mask_visits"]
        flat_m = tf.cast(tf.reshape(m, shape=[-1]), dtype=tf.int32)
        feat_e1 = self.model.E1(x, training=False)
        mean_e1, logvar_e1 = self.model.encode(feat_e1)
        z_e1 = self.model.reparameterize(mean_e1, logvar_e1)
        latent_e1 = tf.concat((z_e1, pe[:,0,:]), axis=-1)
        
        out_de = self.model.De(latent_e1, training=False)
        #dec_loss = self.decode_loss_func(x, out_de, z_e1, mean_e1, logvar_e1, mode='mean')
        dec_loss = self.decode_loss_func(x, out_de, mean_e1, logvar_e1)
        
        out_gens  = []
        out_reals = []
        cons_loss = 0.
        for i in range(self.timepoints):   
            if i==0:
                out_gen = self.model.G1(latent_e1, training=False)
            else:
                feat_y = y[:,i-1,:,:,:,:]
                feat_e2 = self.model.E2(feat_y, training=False)
                mean_e2, logvar_e2 = self.model.encode(feat_e2)
                z_e2 = self.model.reparameterize(mean_e2, logvar_e2)
                latent_e2 = tf.concat((z_e1+z_e2, pe[:,i,:]), axis=-1)
                out_gen = self.model.G1(latent_e2, training=False)
                
                cons_loss += self.CL_func.contrastive_loss(z_e1, z_e2)
                
                out_reals.append(feat_y)
            out_gens.append(out_gen)
        out_reals.append( y[:,-1,:,:,:,:])
        out_reals = tf.concat(out_reals, axis=0)
        out_gens  = tf.concat(out_gens, axis=0)
        
        out_reals = tf.boolean_mask(out_reals, flat_m, axis=0)
        out_gens  = tf.boolean_mask(out_gens, flat_m, axis=0)
        
        real_logits = self.model.Di(out_reals, training=False)
        fake_logits = self.model.Di(out_gens, training=False)
        
        gen_loss = self.gen_loss_func(fake_logits) + self.mse_loss_func(out_reals, out_gens)
        dis_loss = self.dis_loss_func(real_logits, fake_logits)
        
        return {
            'dec_loss':dec_loss,
            'cons_loss':cons_loss,
            'gen_loss':gen_loss,
            'dis_loss':dis_loss,
        }     
            
    def evaluate(self, epoch):
        starttime = time.time()
        
        #self.val_metrics.reset_states()
        
        for x_val, y_val in self.val_ds:
            val_loss = self.evaluate_one_step(x_val, y_val)

            # Update val metrics
            #self.val_loss_metric.update_state(sum(val_loss.values()))
            self.val_loss_metric['dec_loss'].update_state(val_loss['dec_loss'])
            self.val_loss_metric['cons_loss'].update_state(val_loss['cons_loss'])
            self.val_loss_metric['gen_loss'].update_state(val_loss['gen_loss'])
            self.val_loss_metric['dis_loss'].update_state(val_loss['dis_loss'])
            #self.val_metrics.update_state(y_val, val_logits)

        val_dec_loss = self.val_loss_metric['dec_loss'].result()
        val_cons_loss = self.val_loss_metric['cons_loss'].result()
        val_gen_loss = self.val_loss_metric['gen_loss'].result()
        val_dis_loss = self.val_loss_metric['dis_loss'].result()
        # if tf.less(val_loss, self.best_val_loss):
        #     self.save_model(f'epoch_{epoch}')
        #     self.best_val_loss = val_loss

        self.val_loss_metric['dec_loss'].reset_states()
        self.val_loss_metric['cons_loss'].reset_states()
        self.val_loss_metric['gen_loss'].reset_states()
        self.val_loss_metric['dis_loss'].reset_states()
        
        #val_results = self.val_metrics.result()

        print(f"Validation - epoch {epoch}: dec_loss = {val_dec_loss:.3f} -- cons_loss = {val_cons_loss:.3f} -- gen_loss = {val_gen_loss:.3f} -- dis_loss = {val_dis_loss:.3f} -- {time.time()-starttime:.3f} sec")
        
        self.save_model(f"epoch_{epoch}")
    
    def save_model(self, save_name):
        self.model.E1.save_weights(self.model_dir+'/'+save_name+'_E1')
        self.model.E2.save_weights(self.model_dir+'/'+save_name+'_E2')
        self.model.G1.save_weights(self.model_dir+'/'+save_name+'_G1')
        self.model.De.save_weights(self.model_dir+'/'+save_name+'_De')
        self.model.Di.save_weights(self.model_dir+'/'+save_name+'_Di')
        print(f"Save model weights to {self.model_dir}/{save_name}\n")
    
    def load_model(self, save_name):
        self.model.E1.load_weights(self.model_dir+'/'+save_name+'_E1')
        self.model.E2.load_weights(self.model_dir+'/'+save_name+'_E2')
        self.model.G1.load_weights(self.model_dir+'/'+save_name+'_G1')
        self.model.De.load_weights(self.model_dir+'/'+save_name+'_De')
        self.model.Di.load_weights(self.model_dir+'/'+save_name+'_Di')
        print(f"Load model weights from {self.model_dir}/{save_name}\n")
    
    def get_result_metrics(self, dataset):
        psnr_scores, ssim_scores = [], []
        for feat, label in dataset:
            x  = feat["mri_image"]
            pe = feat["pos_enc"]
            y  = label["sal_image"]
            m  = label["mask_visits"]
            
            flat_m = tf.cast(tf.reshape(m, shape=[-1]), dtype=tf.int32)
            feat_e1 = self.model.E1(x, training=False)
            mean_e1, logvar_e1 = self.model.encode(feat_e1)
            z_e1 = self.model.reparameterize(mean_e1, logvar_e1)
            latent_e1 = tf.concat((z_e1, pe[:,0,:]), axis=-1)
            
            out_gens  = []
            out_reals = []
            for i in range(self.timepoints):   
                if i==0:
                    out_gen = self.model.G1(latent_e1, training=True)
                else:
                    feat_y = y[:,i-1,:,:,:,:]
                    feat_e2 = self.model.E2(feat_y, training=True)
                    mean_e2, logvar_e2 = self.model.encode(feat_e2)
                    z_e2 = self.model.reparameterize(mean_e2, logvar_e2)
                    latent_e2 = tf.concat((z_e1+z_e2, pe[:,i,:]), axis=-1)
                    out_gen = self.model.G1(latent_e2, training=True)
                    
                    out_reals.append(feat_y)
                out_gens.append(out_gen)
            out_reals.append( y[:,-1,:,:,:,:])
            out_reals = tf.concat(out_reals, axis=0)
            out_gens  = tf.concat(out_gens, axis=0)
            
            out_reals = tf.boolean_mask(out_reals, flat_m, axis=0)
            out_gens  = tf.boolean_mask(out_gens, flat_m, axis=0)
            
            psnr_scores.append(LO.psnr(out_reals, out_gens))
            #ssim_scores.append(LO.ssim(out_reals, out_gens))
        
        return {
            'psnr': tf.reduce_mean(psnr_scores),
            #'ssim': tf.reduce_mean(ssim_scores)
        }
        
def generate_positional_encoding(length_timepoint, channels):
    def get_emb(sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
        emb = tf.reshape(emb, (*emb.shape[:-2], -1))
        return emb
    
    channels = int(np.ceil(channels / 2) * 2)
    inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, channels, 2) / np.float32(channels)
            )
        )
    dtype = inv_freq.dtype
    pos_x = tf.range(length_timepoint, dtype=dtype)
    sin_inp_x = tf.einsum("i,j->ij", pos_x, inv_freq)
    emb = tf.expand_dims(get_emb(sin_inp_x), 0)
    emb = emb[0]
    return emb

def normalize_minmax(tensor):
    norm_tensor = tf.math.divide_no_nan(
       tf.subtract(
          tensor, 
          tf.reduce_min(tensor)
       ), 
       tf.subtract(
          tf.reduce_max(tensor), 
          tf.reduce_min(tensor)
       )
    )
    return norm_tensor

@tf.numpy_function(Tout=tf.float32)
def random_select_and_resize(input_tensor, feature_map):
    batch, height, width, depth, c = input_tensor.shape
    _, feat_height, feat_width, feat_depth, f = feature_map.shape
    
    if f < c:
        raise ValueError("Number of channels in feature map must be greater than or equal to c.")
    
    selected_slices = np.random.choice(f, c, replace=False)
    selected_feature = feature_map[:, :, :, :, selected_slices]
    
    depth_factor = depth / feat_depth
    width_factor = width / feat_width
    height_factor = height / feat_height
    
    resized_feature = np.zeros_like(input_tensor)

    for i in range(batch):
        resized_feature[i,:,:,:,:] = ndimage.zoom(selected_feature[i, :, :, :, :], (height_factor, width_factor, depth_factor, 1), order=2)
    
    return tf.convert_to_tensor(resized_feature, dtype=tf.float32)