# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# Patch for broken CTRL+C handler
# https://github.com/ContinuumIO/anaconda-issues/issues/905
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import torch, json, numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pathlib import Path
from os import makedirs
from PIL import Image
from netdissect import proggan, nethook, easydict, zdataset
from netdissect.modelconfig import create_instrumented_model
from estimators import get_estimator
from models import get_instrumented_model
from scipy.cluster.vq import kmeans
import re
import sys
import datetime
import argparse
from tqdm import trange
from config import Config
from decomposition import get_random_dirs, get_or_compute, get_max_batch_size, SEED_VISUALIZATION
from utils import pad_frames
from notebooks.notebook_utils import sample_manifold
from PIL import Image


def x_closest(p):
    distances = np.sqrt(np.sum((X - p)**2, axis=-1))
    idx = np.argmin(distances)
    return distances[idx], X[idx]


def make_grid(latent, lat_mean, lat_comp, lat_stdev, act_mean, act_comp, act_stdev, scale=1, n_rows=10, n_cols=5, make_plots=True, edit_type='latent'):
    from notebooks.notebook_utils import create_strip_centered

    inst.remove_edits()
    x_range = np.linspace(-scale, scale, n_cols, dtype=np.float32) # scale in sigmas

    rows = []
    for r in range(n_rows):
        curr_row = []
        out_batch = create_strip_centered(inst, edit_type, layer_key, [latent],
            act_comp[r], lat_comp[r], act_stdev[r], lat_stdev[r], act_mean, lat_mean, scale, 0, -1, n_cols)[0]
        for i, img in enumerate(out_batch):
            curr_row.append(('c{}_{:.2f}'.format(r, x_range[i]), img))

        rows.append(curr_row[:n_cols])

    inst.remove_edits()
    
    if make_plots:
        # If more rows than columns, make several blocks side by side
        n_blocks = 2 if n_rows > n_cols else 1
        
        for r, data in enumerate(rows):
            # Add white borders
            imgs = pad_frames([img for _, img in data]) 
            
            coord = ((r * n_blocks) % n_rows) + ((r * n_blocks) // n_rows)
            plt.subplot(n_rows//n_blocks, n_blocks, 1 + coord)
            plt.imshow(np.hstack(imgs))
            
            # Custom x-axis labels
            W = imgs[0].shape[1] # image width
            P = imgs[1].shape[1] # padding width
            locs = [(0.5*W + i*(W+P)) for i in range(n_cols)]
            plt.xticks(locs, ["{:.2f}".format(v) for v in x_range])
            plt.yticks([])
            plt.ylabel(f'C{r}')

        plt.tight_layout()
        plt.subplots_adjust(top=0.96) # make room for suptitle

    return [img for row in rows for img in row]


######################
### Visualize results
######################

if __name__ == '__main__':
    global max_batch, sample_shape, feature_shape, inst, args, layer_key, model

    args = Config().from_args()
    t_start = datetime.datetime.now()
    timestamp = lambda : datetime.datetime.now().strftime("%d.%m %H:%M")
    print(f'[{timestamp()}] {args.model}, {args.layer}, {args.estimator}')

    # Ensure reproducibility
    torch.manual_seed(0) # also sets cuda seeds
    np.random.seed(0)

    # Speed up backend
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_grad_enabled(False)

    has_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if has_gpu else 'cpu')
    layer_key = args.layer
    layer_name = layer_key #layer_key.lower().split('.')[-1]

    basedir = Path(__file__).parent.resolve()
    outdir = basedir / 'out'

    # Load model
    inst = get_instrumented_model(args.model, args.output_class, layer_key, device, use_w=args.use_w)
    model = inst.model
    feature_shape = inst.feature_shape[layer_key]
    latent_shape = model.get_latent_shape()
    print('Feature shape:', feature_shape)

    # Layout of activations
    if len(feature_shape) != 4: # non-spatial
        axis_mask = np.ones(len(feature_shape), dtype=np.int32)
    else:
        axis_mask = np.array([0, 1, 1, 1]) # only batch fixed => whole activation volume used

    # Shape of sample passed to PCA
    sample_shape = feature_shape*axis_mask
    sample_shape[sample_shape == 0] = 1

    # Load or compute components
    dump_name = get_or_compute(args, inst)
    data = np.load(dump_name, allow_pickle=False) # does not contain object arrays
    X_comp = data['act_comp']
    X_global_mean = data['act_mean']
    X_stdev = data['act_stdev']
    X_var_ratio = data['var_ratio']
    X_stdev_random = data['random_stdevs']
    Z_global_mean = data['lat_mean']
    Z_comp = data['lat_comp']
    Z_stdev = data['lat_stdev']
    n_comp = X_comp.shape[0]
    data.close()

    # Transfer components to device
    tensors = SimpleNamespace(
        X_comp = torch.from_numpy(X_comp).to(device).float(), #-1, 1, C, H, W
        X_global_mean = torch.from_numpy(X_global_mean).to(device).float(), # 1, C, H, W
        X_stdev = torch.from_numpy(X_stdev).to(device).float(),
        Z_comp = torch.from_numpy(Z_comp).to(device).float(),
        Z_stdev = torch.from_numpy(Z_stdev).to(device).float(),
        Z_global_mean = torch.from_numpy(Z_global_mean).to(device).float(),
    )

    transformer = get_estimator(args.estimator, n_comp, args.sparsity)
    tr_param_str = transformer.get_param_str()

    # Compute max batch size given VRAM usage
    max_batch = args.batch_size or (get_max_batch_size(inst, device) if has_gpu else 1)
    print('Batch size:', max_batch)

    def show():
        if args.batch_mode:
            plt.close('all')
        else:
            plt.show()

    print(f'[{timestamp()}] Creating visualizations')

    # Ensure visualization gets new samples
    torch.manual_seed(SEED_VISUALIZATION)
    np.random.seed(SEED_VISUALIZATION)

    # Make output directories
    est_id = f'spca_{args.sparsity}' if args.estimator == 'spca' else args.estimator
    if args.end_c < 0:
        outdir_manifold = outdir/model.name/layer_key.lower()/est_id/'manifold_{}_{}_{}'.format(args.output_class, args.c1, args.c2)
        comp_nums = [args.c1, args.c2]
    else:
        outdir_manifold = outdir/model.name/layer_key.lower()/est_id/'manifold_{}_{}'.format(args.output_class, args.end_c)
        comp_nums = list(range(args.end_c))

    makedirs(outdir_manifold, exist_ok=True)

    # Measure component sparsity (!= activation sparsity)
    sparsity = np.mean(X_comp == 0) # percentage of zero values in components
    print(f'Sparsity: {sparsity:.2f}')


    if args.gen_grid:
        n = 6
        assert len(comp_nums) == 2
        print('Generating a manifold grid image...')
        mosaic = []
        for s0 in np.arange(-args.sigma, args.sigma+1e-3, args.sigma/n):
            manifold_latents = []
            for s1 in np.arange(-args.sigma, args.sigma+1e-3, args.sigma/n):
                manifold_latents.append([s0, s1])
            manifold_latents = np.array(manifold_latents).astype(np.float32)
            frames, latents = sample_manifold(inst,
                                              tensors.Z_global_mean,
                                              tensors.Z_comp[comp_nums], tensors.Z_stdev[comp_nums], args.sigma, 0, -1,
                                              manifold_latents.shape[0], manifold_latents=manifold_latents)
            mosaic.append(np.hstack(frames))
        mosaic = np.vstack(mosaic)
        # frame = np.vstack([np.hstack([frames[i*5+j] for j in range(5)]) for i in range(5)])
        im = Image.fromarray((255*mosaic).astype(np.uint8))
        im = im.resize((128*(2*n+1), 128*(2*n+1)), Image.BILINEAR)
        print('Saving to', f'{outdir_manifold}_grid.jpg')
        im.save(f'{outdir_manifold}_grid.jpg')

    else:
        # Random center:
        # latents = model.sample_latent(n_samples=1)
        # z = latents[0][None, ...]

        print('Sampling from manifold...')
        frames, latents = sample_manifold(inst,
                                          tensors.Z_global_mean,
                                          tensors.Z_comp[comp_nums], tensors.Z_stdev[comp_nums], args.sigma, 0, -1,
                                          args.samples)

        for i in range(len(frames)):
            im = Image.fromarray((255*frames[i]).astype(np.uint8))
            im = im.resize((256, 256), Image.BILINEAR)
            im.save(f'{outdir_manifold}/{i:06d}.jpg')
        np.savetxt(f'{outdir_manifold}/latents.txt', latents, '%.8f')
    print('Done in', datetime.datetime.now() - t_start)
