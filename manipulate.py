"""Manipulates real images with LIA with given boundary."""

import os
import sys
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from utils import imwrite, immerge
from utils import preparing_data
from training.misc import load_pkl
import dnnlib
import dnnlib.tflib as tflib


def manipulate(latent_code,
               boundary,
               num_layers=12,
               step=11,
               start_distance=-5.,
               end_distance=5.):
    """Manipulates the given latent code with respect to a particular boundary.
    Basically, this function takes a latent code and a boundary as inputs, and
    outputs a collection of manipulated latent codes. For example, let `steps` to
    be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
    `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
    with shape [10, num_layers, latent_space_dim].
    NOTE: Distance is sign sensitive.
    Args:
      latent_code: The input latent code for manipulation.
      boundary: The semantic boundary as reference.
      num_layers: Number of layers to repeat the code.
      start_distance: The distance to the boundary where the manipulation starts.
        (default: -5.0)
      end_distance: The distance to the boundary where the manipulation ends.
        (default: 5.0)
      steps: Number of steps to move the latent code from start position to end
        position. (default: 11)
    Returns:
      Interpolated code.
    """
    assert (len(latent_code.shape) == 2 and len(boundary.shape) == 2 and
            latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
            latent_code.shape[1] == boundary.shape[1])

    linspace = np.linspace(start_distance, end_distance, step)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    replaced_code = (latent_code + linspace * boundary)[:, np.newaxis]
    repeated_code = np.tile(replaced_code, [1, num_layers, 1])

    return repeated_code


def parse_args():
    """Parses arguments."""
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()

    parser.add_argument('--restore_path', type=str, default='',
                        help='The pre-trained encoder pkl file path')
    parser.add_argument("--data_dir_test", type=str, default='',
                        help="Location of the test data")
    parser.add_argument("--img_type", type=str, default='.png',
                        help="test images type, such as .jpg., .png")
    parser.add_argument("--boundary", type=str, default='',
                        help="Location of the boundary file")
    parser.add_argument("--image_size", type=int,
                        default=128, help="the training image size")
    parser.add_argument("--batch_size", type=int,
                        default=8, help="size of the input batch")
    parser.add_argument('--start_distance', type=float, default=-5.0,
                        help='Start distance for manipulation. (default: -5.0)')
    parser.add_argument('--end_distance', type=float, default=5.0,
                        help='End distance for manipulation. (default: 5.0)')
    parser.add_argument("--step", type=int,
                        default=8, help="manipulation total steps")
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified, '
                             '`./outputs/manipulation` will be used by default.')
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
    assert os.path.exists(args.boundary)
    E, _, _, Gs, _ = load_pkl(args.restore_path)
    num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]

    # Building graph
    real = tf.placeholder('float32', [None, 3, args.image_size, args.image_size], name='real_image')
    W = tf.placeholder('float32', [None, num_layers, latent_dim], name='Gaussian')
    encoder_w = E.get_output_for(real, phase=False)
    reconstruction_from_w = Gs.components.synthesis.get_output_for(W, randomize_noise=False)
    sess = tf.get_default_session()

    # Preparing data
    input_images, images_name = preparing_data(im_path=args.data_dir_test, img_type=args.img_type)

    boundary = np.load(args.boundary)
    boundary_name = args.boundary.split('/')[-1].split('_')[0]

    save_dir = args.output_dir or './outputs/manipulation'
    os.makedirs(save_dir, exist_ok=True)

    print('manipulation in w space on %s' % (boundary_name))
    for i in tqdm(range(input_images.shape[0])):
        input_image = input_images[i:i+1]
        im_name = images_name[i]
        latent_code = sess.run(encoder_w, feed_dict={real: input_image})
        codes = manipulate(latent_code,
                           boundary,
                           num_layers=num_layers,
                           step=args.step,
                           start_distance=args.start_distance,
                           end_distance=args.end_distance)
        inputs = np.zeros((args.batch_size, num_layers, latent_dim), np.float32)
        output_images = []
        for idx in range(0, args.step, args.batch_size):
            batch = codes[idx:idx + args.batch_size]
            inputs[0:len(batch)] = batch
            images = sess.run(reconstruction_from_w, feed_dict={W: inputs})
            output_images.append(images[0:len(batch)])
        output_images = np.concatenate(output_images, axis=0)
        output_images = np.concatenate([input_image, output_images], axis=0)
        output_images = output_images.transpose(0, 2, 3, 1)
        imwrite(immerge(output_images, 1, args.step + 1), '%s/%s_%s.png' %
                (save_dir, im_name, boundary_name))



if __name__ == "__main__":
    main()
