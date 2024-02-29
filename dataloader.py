import os
import ast
#import glob
#import math
import numpy as np
import pandas as pd
import tensorflow as tf

from nilearn import image
#from skimage import transform
from layers import TFPositionalEncoding3D

from typing import Dict, Iterable, Tuple, List

def normalize_image(image):
    return (image - image.min()) / (image.max() -image.min() + 1e-6)

def resize(img, new_shape, interpolation="nearest"):
    input_shape = np.asarray(img.shape, dtype=np.float16)
    ras_image = image.reorder_img(img, resample=interpolation)
    output_shape = np.asarray(new_shape)
    new_spacing = input_shape/output_shape
    new_affine = np.copy(ras_image.affine)
    new_affine[:3, :3] = ras_image.affine[:3, :3] * np.diag(new_spacing)
    return image.resample_img(ras_image, target_affine=new_affine, target_shape=output_shape, interpolation=interpolation)

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

class InputFunction:
    def __init__(self,
                 filepath  : str,
                 list_rids : list,
                 num_classes : int,
                 labeltime   : str,
                 image_shape : tuple = (84,48,42),
                 img_normalize : bool = False,
                 use_pe     : bool = False,
                 augmented  : bool = False,
                 timesteps  : int  = 10,
                 batch_size : int  = 4,
                 drop_last  : bool = False,
                 shuffle    : bool = False,
                 seed       : int  = 2023) -> None:
        self.filepath    = filepath
        self.list_rids   = list_rids
        self.labeltime   = labeltime
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.img_normalize = img_normalize
        self.use_pe     = use_pe
        self.augmented  = augmented
        self.timesteps  = timesteps
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.shuffle    = shuffle
        self.seed       = seed
        
        df = pd.read_csv(self.filepath)
        self.df = df.query('RID in @self.list_rids')
        self.df_neg_samples = self._get_negative_samples()
        
    def size(self) -> int:
        return len(self.df)
    
    def steps_per_epoch(self) -> int:
        return int(np.floor(self.size() / self.batch_size))
    
    def select_label_time(self, path) -> List[str]:
        selected_path = []
        follow_year = int(self.labeltime[0]) * 12
        for  p in path:
            sr_visit, ds_visit = p.split("/")[-1].split("_")
            if abs(int(ds_visit) - int(sr_visit)) <= follow_year:
                selected_path.append(p)
        return selected_path
    
    def _get_negative_samples(self):
        df_neg = {}
        df_neg['neg_1'] = self.df[self.df[f'CONV_STATE_{self.labeltime}']==1]
        df_neg['neg_2'] = self.df[self.df[f'CONV_STATE_{self.labeltime}']==2]
        
        return df_neg
    
    def _compute_positional_encoding(self, feature, timepoint) -> np.ndarray:
        p_enc_3d = TFPositionalEncoding3D(timepoint)
        feature  = tf.convert_to_tensor(feature[None,:,:,:,None], dtype=tf.float32)
        pe = p_enc_3d(feature)
        pe_z = feature + 0.1*pe
        return pe_z[0,...,0]

    def _generate_feature_dic(self, paths, key_features=None) -> Dict[str, np.ndarray]:
        features    = {key: np.zeros((len(paths),)+self.image_shape+(self.timesteps,)) for key in key_features}
        delta_feats = np.zeros((len(paths), self.timesteps))
        mask_visits = np.zeros((len(paths), self.timesteps))
        keep_list = []
        for i, p in enumerate(paths):
            p = ast.literal_eval(p)
            p = self.select_label_time(p)
            if p:
                num_visits = len(p)
                start_index = self.timesteps - num_visits
                heads, tails = zip(*(os.path.split(path) for path in p))
                path_tuples = list(zip(heads, tails))
                sorted_path_tuples = sorted(path_tuples, key=lambda x: x[1])
                #sorted_heads, sorted_tails = zip(*sorted_path_tuples)
                for j, (h, t) in enumerate(sorted_path_tuples):
                    bl_visit = t.split("_")[0]
                    tg_visit = t.split("_")[1]
                    delta = int(tg_visit) - int(bl_visit)
                    delta_feats[i,j+start_index] = delta
                    mask_visits[i,j+start_index] = 1.
                    image_ad = image.load_img(os.path.join(h, t, 'crop_salmap_ad.nii'))
                    image_mg = image.load_img(os.path.join(h, t, 'crop_salmap_mg.nii'))
                    image_dg = image.load_img(os.path.join(h, t, 'crop_salmap_dg_1.nii'))
                    
                    image_ad_resized = resize(image_ad, self.image_shape)
                    image_mg_resized = resize(image_mg, self.image_shape)
                    image_dg_resized = resize(image_dg, self.image_shape)
                    
                    image_ad_resized = image_ad_resized.get_fdata()
                    image_mg_resized = image_mg_resized.get_fdata()
                    image_dg_resized = image_dg_resized.get_fdata()
                    
                    if self.use_pe:
                        image_ad_resized = self._compute_positional_encoding(image_ad_resized, delta)
                        image_mg_resized = self._compute_positional_encoding(image_mg_resized, delta)
                        image_dg_resized = self._compute_positional_encoding(image_dg_resized, delta)
                    
                    features[key_features[0]][i,...,j+start_index] = image_ad_resized
                    features[key_features[1]][i,...,j+start_index] = image_mg_resized
                    features[key_features[2]][i,...,j+start_index] = image_dg_resized
                    
                keep_list.append(i)
        features[key_features[0]] = tf.cast(tf.gather(features[key_features[0]], indices=keep_list, axis=0), dtype=tf.float32)
        features[key_features[1]] = tf.cast(tf.gather(features[key_features[1]], indices=keep_list, axis=0), dtype=tf.float32)
        features[key_features[2]] = tf.cast(tf.gather(features[key_features[2]], indices=keep_list, axis=0), dtype=tf.float32)
        features['deltas'] = tf.cast(tf.gather(delta_feats, indices=keep_list, axis=0), dtype=tf.float32)
        features['mask'] = tf.cast(tf.gather(mask_visits, indices=keep_list, axis=0), dtype=tf.float32)
        
        return features, keep_list
        

    def _get_image_from_path(self, index: np.ndarray) -> Dict[str, tf.Tensor]:
        selected_rows = self.df.iloc[index]
        paths = selected_rows['SAL_PATHS'].tolist()
        
        features, keep_list = self._generate_feature_dic(paths, key_features=['AbsDiff','MagGrad','DirGrad'])
        
        if self.augmented:
            neg_rows_1 = self.df_neg_samples['neg_1'].sample(int(len(paths)))
            neg_rows_2 = self.df_neg_samples['neg_2'].sample(int(len(paths)))
            neg_paths_1 = neg_rows_1['SAL_PATHS'].tolist()
            neg_paths_2 = neg_rows_2['SAL_PATHS'].tolist()
            
            neg_features_1, neg_keep_list_1 = self._generate_feature_dic(neg_paths_1, key_features=['AbsDiff_neg_1','MagGrad_neg_1','DirGrad_neg_1'])
            neg_features_2, neg_keep_list_2 = self._generate_feature_dic(neg_paths_2, key_features=['AbsDiff_neg_2','MagGrad_neg_2','DirGrad_neg_2'])
            
            merged_features = {}
            all_features = [features, neg_features_1, neg_features_2]
            for feature_dict in all_features:
                for key, value in feature_dict.items():
                    if key in merged_features:
                        # If the key already exists, concatenate values along axis 0
                        merged_features[key] = tf.concat([merged_features[key], value], axis=0)
                    else:
                        # If the key is not present, add it to the merged dictionary
                        merged_features[key] = value
            
            return merged_features, [keep_list, neg_keep_list_1, neg_keep_list_2]
        else:
            return features, keep_list

    
    def _get_labels(self, index: np.ndarray, keep_list: list) -> Dict[str, tf.Tensor]:
        selected_rows = self.df.iloc[index]
        lb_array = selected_rows[f'CONV_STATE_{self.labeltime}'].values
        labels = {'conv_lb' : self.one_hot_encoding(lb_array, num_classes=self.num_classes)}
        if self.augmented:
            labels['conv_lb'] = tf.cast(tf.gather(labels['conv_lb'], indices=keep_list[0], axis=0), dtype=tf.float32)
            
            neg_arr_1 = np.zeros((lb_array.shape[0], self.num_classes))
            neg_arr_1[:, 1] = 1.
            labels['conv_lb_neg1'] = tf.cast(tf.gather(neg_arr_1, indices=keep_list[1], axis=0), dtype=tf.float32)
            
            neg_arr_2 = np.zeros((lb_array.shape[0], self.num_classes))
            neg_arr_2[:, 2] = 1.
            labels['conv_lb_neg2'] = tf.cast(tf.gather(neg_arr_2, indices=keep_list[2], axis=0), dtype=tf.float32)
            
        else:
            labels['conv_lb'] = tf.cast(tf.gather(labels['conv_lb'], indices=keep_list, axis=0), dtype=tf.float32)
        
        return labels
        
    def one_hot_encoding(self, array: np.ndarray, num_classes: int) -> np.ndarray:
        input_tensor = tf.constant(array, dtype=tf.int32)
        encoded_data = tf.one_hot(input_tensor, depth=num_classes)
        return encoded_data.numpy()
    
    def _get_data_batch(self, index: np.ndarray) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        features, keep_list = self._get_image_from_path(index)
        labels   = self._get_labels(index, keep_list)
        
        return features, labels
    
    def _iter_data(self) -> Iterable[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]]:
        index = np.arange(self.size())
        rnd = np.random.RandomState(self.seed)
        
        if self.shuffle:
            rnd.shuffle(index)
        for b in range(self.steps_per_epoch()):
            start = b * self.batch_size
            idx = index[start:(start + self.batch_size)]
            yield self._get_data_batch(idx)

        if not self.drop_last:
            start = self.steps_per_epoch() * self.batch_size
            idx = index[start:]
            if not idx.size==0:
                yield self._get_data_batch(idx)
    
    def _get_signature(self) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        if self.augmented:
            features = {
                'AbsDiff': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'MagGrad': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'DirGrad': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'AbsDiff_neg_1': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'MagGrad_neg_1': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'DirGrad_neg_1': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'AbsDiff_neg_2': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'MagGrad_neg_2': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'DirGrad_neg_2': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'deltas' : tf.TensorSpec(shape=(None, self.timesteps), dtype=tf.float32),
                'mask'   : tf.TensorSpec(shape=(None, self.timesteps), dtype=tf.float32),
            }
            labels = {
                'conv_lb': tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32),
                'conv_lb_neg1': tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32),
                'conv_lb_neg2': tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32),
            }
        else:
            features = {
                'AbsDiff': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'MagGrad': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'DirGrad': tf.TensorSpec(shape=(None,)+self.image_shape+(self.timesteps,), dtype=tf.float32),
                'deltas' : tf.TensorSpec(shape=(None, self.timesteps), dtype=tf.float32),
                'mask'   : tf.TensorSpec(shape=(None, self.timesteps), dtype=tf.float32),
            }
            labels = {
                'conv_lb': tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.float32),
            }
        output_signature = (features, labels)
        return output_signature
    
    def _make_dataset(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            self._iter_data,
            output_signature = self._get_signature()
        )
        return ds
    
    def __call__(self) -> tf.data.Dataset:
        return self._make_dataset()

class InputFunctionS2:
    def __init__(self,
                 source_dir: str,
                 filepath  : str,
                 list_rids : list,
                 input_name: str,
                 img_shape : tuple = (189,216,189),
                 sal_shape : tuple = (84,48,42),
                 pe_dim    : int = 32,
                 img_normalize : bool = False,
                 segment_input : bool = False,
                 timesteps  : int  = 10,
                 batch_size : int  = 4,
                 drop_last  : bool = False,
                 shuffle    : bool = False,
                 seed       : int  = 2024) -> None:
        self.source_dir = source_dir
        self.filepath   = filepath
        self.list_rids  = list_rids
        self.input_name = input_name
        self.img_shape = img_shape
        self.sal_shape = sal_shape
        self.pe_dim    = pe_dim
        self.img_normalize = img_normalize
        self.segment_input = segment_input
        self.timesteps  = timesteps
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.shuffle    = shuffle
        self.seed       = seed
        
        df = pd.read_csv(self.filepath)
        self.df = df.query('RID in @self.list_rids')
        self.pos_enc = generate_positional_encoding(150, self.pe_dim)
        
    def size(self) -> int:
        return len(self.df)
    
    def steps_per_epoch(self) -> int:
        return int(np.floor(self.size() / self.batch_size))  
    
    def _get_data_batch(self, index: np.ndarray) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        selected_rows = self.df.iloc[index]
        img_paths = selected_rows['MRI_IMG'].tolist()
        sal_paths = selected_rows['SAL_PATHS'].tolist()
        
        images      = np.zeros((self.batch_size,)+self.img_shape+(1,))
        sal_images  = np.zeros((self.batch_size, self.timesteps)+self.sal_shape+(1,))
        deltas      = np.zeros((self.batch_size, self.timesteps))
        mask_visits = np.zeros((self.batch_size, self.timesteps))
        pos_enc     = np.zeros((self.batch_size, self.timesteps, self.pe_dim))
        
        for i, sp in enumerate(sal_paths):
            sp = ast.literal_eval(sp)
            if sp:
                #num_visits = len(sp)
                #start_index = self.timesteps - num_visits
                heads, tails = zip(*(os.path.split(path) for path in sp))
                path_tuples = list(zip(heads, tails))
                sorted_path_tuples = sorted(path_tuples, key=lambda x: x[1])
                for j, (h, t) in enumerate(sorted_path_tuples):
                    if j < self.timesteps:
                        bl_visit = t.split("_")[0]
                        tg_visit = t.split("_")[1]
                        delta = int(tg_visit) - int(bl_visit)
                        deltas[i,j] = delta
                        mask_visits[i,j] = 1.
                        pos_enc[i,j,:] = self.pos_enc[int(delta), :]
                        
                        if self.input_name=="AbsDiff":
                            sal_img = image.load_img(os.path.join(self.source_dir, h, t, 'crop_salmap_ad.nii'))
                        elif self.input_name=="MagGrad":
                            sal_img = image.load_img(os.path.join(self.source_dir, h, t, 'crop_salmap_mg.nii'))
                        elif self.input_name=="DirGrad":
                            sal_img = image.load_img(os.path.join(self.source_dir, h, t, 'crop_salmap_dg_1.nii'))
                        sal_img_resized = resize(sal_img, self.sal_shape)
                        sal_img_resized = np.nan_to_num(sal_img_resized.get_fdata())
                        if self.img_normalize:
                            sal_img_resized = normalize_image(sal_img_resized)
                        sal_images[i,j,...,0] = sal_img_resized
        
        for i, ip in enumerate(img_paths):
            img = image.load_img(os.path.join(self.source_dir,ip))
            if self.segment_input:
                curr_id = ip.split('/')[1]
                img_mask = image.load_img(os.path.join(self.source_dir,'ADNI_saliency',curr_id,'Hippocampus','region_mask.nii'))
                
            img = resize(img, self.img_shape)
            img_data = img.get_fdata()
            img_data = np.nan_to_num(img_data)
            if self.img_normalize:
                img_data = normalize_image(img_data)
            images[i,...,0] = img_data
        
        features = {
            "mri_image": images,
            "pos_enc"  : pos_enc,
        }
        
        labels = {
            "sal_image": sal_images,
            "deltas"   : deltas,
            "mask_visits": mask_visits,
        }
        
        return features, labels
    
    def _iter_data(self) -> Iterable[Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]]:
        index = np.arange(self.size())
        rnd = np.random.RandomState(self.seed)
        
        if self.shuffle:
            rnd.shuffle(index)
        for b in range(self.steps_per_epoch()):
            start = b * self.batch_size
            idx = index[start:(start + self.batch_size)]
            yield self._get_data_batch(idx)

        if not self.drop_last:
            start = self.steps_per_epoch() * self.batch_size
            idx = index[start:]
            if not idx.size==0:
                yield self._get_data_batch(idx)
    
    def _get_signature(self) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        features = {
            'mri_image': tf.TensorSpec(shape=(None,)+self.img_shape+(1,), dtype=tf.float32),
            'pos_enc'  : tf.TensorSpec(shape=(None, self.timesteps, self.pe_dim), dtype=tf.float32),
        }
        labels = {
            'sal_image': tf.TensorSpec(shape=(None, self.timesteps)+self.sal_shape+(1,), dtype=tf.float32),
            'deltas'   : tf.TensorSpec(shape=(None, self.timesteps), dtype=tf.float32),
            'mask_visits': tf.TensorSpec(shape=(None, self.timesteps), dtype=tf.float32),
        }
        output_signature = (features, labels)
        return output_signature
    
    def _make_dataset(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            self._iter_data,
            output_signature = self._get_signature()
        )
        return ds
    
    def __call__(self) -> tf.data.Dataset:
        return self._make_dataset()