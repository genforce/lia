"""Synthesizes images with LIA."""

import os
import sys
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from utils import imwrite, immerge
from training.misc import load_pkl
import dnnlib
import dnnlib.tflib as tflib


def parse_args():
    """Parses arguments."""
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()

    parser.add_argument('--restore_path', type=str, default='',
                        help='The pre-trained encoder pkl file path')
    parser.add_argument("--batch_size", type=int,
                        default=8, help="size of the input batch")
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified, '
                             '`./outputs/sampling` will be used by default.')
    parser.add_argument('--total_nums', type=int, default=5,
                        help='number of loops for sampling')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    tf_config = {'rnd.np_random_seed': 1000}
    tflib.init_tf(tf_config)
    assert os.path.exists(args.restore_path)
    _, _, _, Gs, _ = load_pkl(args.restore_path)
    latent_dim = Gs.components.synthesis.input_shape[2]

    # Building graph
    Z = tf.placeholder('float32', [None, latent_dim], name='Gaussian')
    sampling_from_z = Gs.get_output_for(Z, None, randomize_noise=True)
    sess = tf.get_default_session()

    save_dir = args.output_dir or './outputs/sampling'
    os.makedirs(save_dir, exist_ok=True)

    print('Sampling...')
    for it in tqdm(range(args.total_nums)):
        samples = sess.run(sampling_from_z, {Z: np.random.randn(args.batch_size * 2, latent_dim)})
        samples = samples.transpose(0, 2, 3, 1)
        imwrite(immerge(samples, 2, args.batch_size), '%s/sampling_%06d.png' % (save_dir, it))



if __name__ == "__main__":
    main()
