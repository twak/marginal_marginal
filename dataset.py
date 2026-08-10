import os
import json
import numpy as np

sar_c = 0.0001

sar_transformations = {
    '': lambda x: x,
    '_log': lambda x: np.log(np.maximum(x, sar_c) + sar_c),
    '_log10': lambda x: np.log10(np.maximum(x, sar_c) + sar_c) * 10,
    '_limit': lambda x: np.log(np.maximum(np.minimum(x, 0.05), 0.000001) + sar_c)
}

class PinDataset:
    def __init__(self, data_dir, partition='train'):
        self.data_dir = data_dir
        self.partition = partition
        
        npy_path = os.path.join(data_dir, f"{partition}_data.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Data file not found: {npy_path}")
        self.data = np.load(npy_path)
        
        json_path = os.path.join(data_dir, f"{partition}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON metadata not found: {json_path}")
        with open(json_path, 'r') as f:
            self.metadata = json.load(f)
            
        if len(self.data) != len(self.metadata):
            raise ValueError(f"Length mismatch: data ({len(self.data)}) vs metadata ({len(self.metadata)})")
            
        meta_path = os.path.join(data_dir, "mean_std.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Global metadata not found: {meta_path}")
        with open(meta_path, 'r') as f:
            self.global_meta = json.load(f)
            
        self.channels = self.global_meta.get('channels', [])
        self.original_channels = self.channels.copy()
        
        self.means = None
        self.stds = None
        
        if 'means' in self.global_meta and 'std' in self.global_meta:
            meta_channels = self.global_meta.get('channels', [])
            all_means = self.global_meta.get('means', [])
            all_stds = self.global_meta.get('std', [])
            
            means_log = self.global_meta.get('means_log', all_means)
            stds_log = self.global_meta.get('std_log', all_stds)
            
            means_log10 = self.global_meta.get('means_log10', all_means)
            stds_log10 = self.global_meta.get('std_log10', all_stds)
            
            means = []
            stds = []
            new_channels = []
            
            for ch in self.original_channels:
                if ch in meta_channels:
                    idx = meta_channels.index(ch)
                    if ch.lower().startswith("s1:"):
                        means.append(all_means[idx])
                        stds.append(all_stds[idx])
                        new_channels.append(ch)
                        
                        means.append(means_log[idx])
                        stds.append(stds_log[idx])
                        new_channels.append(ch + "_log")
                        
                        means.append(means_log10[idx])
                        stds.append(stds_log10[idx])
                        new_channels.append(ch + "_log10")
                    else:
                        means.append(all_means[idx])
                        stds.append(all_stds[idx])
                        new_channels.append(ch)
                else:
                    means.append(0.0)
                    stds.append(1.0)
                    new_channels.append(ch)
            
            self.means = np.array(means, dtype=np.float32)
            self.stds = np.array(stds, dtype=np.float32)
            self.channels = new_channels
        else:
            new_channels = []
            for ch in self.original_channels:
                if ch.lower().startswith("s1:"):
                    new_channels.extend([ch, ch + "_log", ch + "_log10"])
                else:
                    new_channels.append(ch)
            self.channels = new_channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_orig = self.data[idx].astype(np.float32)
        new_x = []
        for i, ch in enumerate(self.original_channels):
            val = x_orig[i]
            if ch.lower().startswith("s1:"):
                new_x.append(val)
                new_x.append(sar_transformations['_log'](val))
                new_x.append(sar_transformations['_log10'](val))
            else:
                new_x.append(val)
                
        x = np.array(new_x, dtype=np.float32)
        
        if self.means is not None and self.stds is not None:
            x = (x - self.means) / (self.stds + 1e-6)
            
        y_str = self.metadata[idx].get("classification", "unknown")
        oracle = self.metadata[idx].get("oracle_classification", None)
        
        return x, y_str, oracle


def load_data(data_dir, partition, modis_mode='all'):
    dataset = PinDataset(data_dir, partition=partition)
    
    X = []
    y = []
    y_oracle = []
    
    has_oracle = len(dataset.metadata) > 0 and 'oracle_classification' in dataset.metadata[0]
    
    for i in range(len(dataset)):
        features, label, oracle = dataset[i]
        X.append(features)
        y.append(label)
        if has_oracle:
            y_oracle.append(oracle)
            
    X = np.array(X)
    channels = dataset.channels
    
    if modis_mode != 'all':
        keep_idx = []
        for i, f in enumerate(channels):
            if modis_mode == 'warped':
                if not f.startswith("modis:"):
                    keep_idx.append(i)
            elif modis_mode == 'unwarped':
                if not f.startswith("modis_warped:"):
                    keep_idx.append(i)
            elif modis_mode == 'both':
                keep_idx.append(i)
        
        X = X[:, keep_idx]
        channels = [channels[i] for i in keep_idx]
        
    keep_idx_s1 = []
    for i, f in enumerate(channels):
        if f.lower().startswith("s1:"):
            if f.endswith("_log10"):
                keep_idx_s1.append(i)
        else:
            keep_idx_s1.append(i)
            
    X = X[:, keep_idx_s1]
    channels = [channels[i] for i in keep_idx_s1]
        
    return X, np.array(y), np.array(y_oracle) if has_oracle else None, channels
