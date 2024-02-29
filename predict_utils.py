import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import accuracy_score, balanced_accuracy_score,  average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

class Predictor:
    def __init__(self, model, model_dir, input_names, augmented=False, ckpt_name=None):
        self.model       = model
        self.model_dir   = model_dir
        self.input_names = input_names
        self.augmented   = augmented
        self.ckpt_name   = ckpt_name
        self.y_true      = None
        self.y_pred      = None
        
    
    def predict(self, dataset):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            optimizer=tf.keras.optimizers.Adam(),
            model=self.model)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, str(self.model_dir), max_to_keep=5)

        if self.ckpt_name:
            ckpt.restore(f"{self.model_dir}/{self.ckpt_name}").expect_partial()
            print(f"Latest checkpoint restored from {self.model_dir}/{self.ckpt_name}.")
        else:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print(f"Latest checkpoint restored from {ckpt_manager.latest_checkpoint}.")
        
        steps = dataset.steps_per_epoch()
        if (not dataset.drop_last) and (dataset.steps_per_epoch()*dataset.batch_size<dataset.size()):
            steps += 1
            
        y_true, y_pred = [], []
        pbar = tqdm(dataset(), total=steps)

        for data, label in pbar:
            pbar.set_description("Predicting: ", refresh=True)
            if self.augmented:
                logits, _ = self.model(data[self.input_names], training=False)
            else:
                logits = self.model(data, training=False)
            y_true.append(label['conv_lb'].numpy())
            y_pred.append(logits.numpy())
        pbar.close()
        
        self.y_true = np.concatenate(y_true, axis=0)
        self.y_pred = np.concatenate(y_pred, axis=0)
    
    def plot_confusion_matrix(self, columns, cmap='PuRd', save_path=None):
        y_true, y_pred = self.y_true, self.y_pred
        y_true_argmax = np.argmax(y_true, axis=-1)
        y_pred_argmax = np.argmax(y_pred, axis=-1)
        
        pp_matrix_from_data(y_true_argmax, y_pred_argmax, columns=columns, cmap=cmap, save_path=save_path)
    
    def get_performance(self):
        if (self.y_true is not None) and (self.y_pred is not None):
            y_true, y_pred = self.y_true, self.y_pred
            y_true_argmax = np.argmax(y_true, axis=-1)
            y_pred_argmax = np.argmax(y_pred, axis=-1)
            results = {
                "accuracy" : accuracy_score(y_true_argmax, y_pred_argmax),
                "balance_accuracy" : balanced_accuracy_score(y_true_argmax, y_pred_argmax),
                "average_precision" : average_precision_score(y_true, y_pred, average="macro"),
                "f1" : f1_score(y_true_argmax, y_pred_argmax, average="macro"),
                "precision" : precision_score(y_true_argmax, y_pred_argmax, average="macro"),
                "recall"    : recall_score(y_true_argmax, y_pred_argmax, average="macro"),
                "roc_auc"   : roc_auc_score(y_true, y_pred, average="macro"),
            }
            return results
        else:
            raise Exception("Run predictor.predict() first!")
        
    
        
        