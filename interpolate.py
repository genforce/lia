"""Interpolates real images with LIA """
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


def linear_interpolate(src_code, dst_code, step=5):
    """Interpolates two latent codes linearlly.
    Args:
      src_code: Source code, with shape [1, latent_space_dim].
      dst_code: Target code, with shape [1, latent_space_dim].
      step: Number of interploation steps. (default: 5)
    Returns:
      Interpolated code, with shape [step, latent_space_dim].
    """
    assert (len(src_code.shape) == 2 and len(dst_code.shape) == 2 and
            src_code.shape[0] == 1 and dst_code.shape[0] == 1 and
            src_code.shape[1] == dst_code.shape[1])

    linspace = np.linspace(0.0, 1.0, step)[:, np.newaxis].astype(np.float32)
    return src_code + linspace * (dst_code - src_code)


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
    parser.add_argument("--image_size", type=int,
                        default=128, help="the training image size")
    parser.add_argument("--batch_size", type=int,
                        default=8, help="size of the input batch")
    parser.add_argument("--step", type=int,
                        default=7, help="interpolation steps between two images")
    parser.add_argument("--mode", type=int,
                        default=1, help="to choose on which space to perfrom"
                                        " interpolation, 0==>z, 1==>w")
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified, '
                             '`./outputs/interpolation_on_w/z` will be used by default.')
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
    E, _, _, Gs, NE = load_pkl(args.restore_path)
    num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]

    # Building graph
    # Interpolation on z
    real = tf.placeholder('float32', [None, 3, args.image_size, args.image_size], name='real_image')
    Z = tf.placeholder('float32', [None, latent_dim], name='Gaussian')
    latent_w = E.get_output_for(real, phase=False)
    latent_z = NE.get_output_for(latent_w, None)
    reconstruction_from_z = Gs.get_output_for(Z, None, randomize_noise=False, is_validation=True)
    # Interpolation on w
    W = tf.tile(Z[:, np.newaxis], [1, num_layers, 1])
    reconstruction_from_w = Gs.components.synthesis.get_output_for(W, randomize_noise=False)
    sess = tf.get_default_session()

    # Using ND's weights to setting NE's weights (invertible)
    ND_vars_pairs = {name: tflib.run(val) for name, val in Gs.components.mapping.vars.items()}
    for ne_name, ne_val in NE.vars.items():
        tflib.set_vars({ne_val: ND_vars_pairs[ne_name]})

    # Preparing data
    input_images, images_name = preparing_data(im_path=args.data_dir_test, img_type=args.img_type)

    if args.mode == 0:
        save_dir = args.output_dir or './outputs/interpolation_on_z'
        os.makedirs(save_dir, exist_ok=True)

        print('Interpolation on z space...')
        for i in tqdm(range(input_images.shape[0])):
            source_image = input_images[i:i+1]
            source_name = images_name[i]
            for j in range(input_images.shape[0]):
                target_image = input_images[j:j + 1]
                target_name = images_name[j]
                source_code = sess.run(latent_z, feed_dict={real: source_image})
                target_code = sess.run(latent_z, feed_dict={real: target_image})

                codes = linear_interpolate(src_code=source_code,
                                           dst_code=target_code,
                                           step=args.step)

                inputs = np.zeros((args.batch_size, latent_dim), np.float32)
                output_images = []
                for idx in range(0, args.step, args.batch_size):
                    batch = codes[idx:idx + args.batch_size]
                    inputs[0:len(batch)] = batch
                    images = sess.run(reconstruction_from_z, feed_dict={Z: inputs})
                    output_images.append(images[0:len(batch)])
                output_images = np.concatenate(output_images, axis=0)
                final_results = np.concatenate([source_image, output_images, target_image], axis=0)
                final_results = final_results.transpose(0, 2, 3, 1)
                imwrite(immerge(final_results, 1, args.step + 2), '%s/%s_to_%s.png' %
                        (save_dir, source_name, target_name))

    elif args.mode == 1:
        save_dir = args.output_dir or './outputs/interpolation_on_w'
        os.makedirs(save_dir, exist_ok=True)

        print('Interpolation on w space...')
        for i in tqdm(range(input_images.shape[0])):
            source_image = input_images[i:i+1]
            source_name = images_name[i]
            for j in range(input_images.shape[0]):
                target_image = input_images[j:j + 1]
                target_name = images_name[j]
                source_code = sess.run(latent_w, feed_dict={real: source_image})
                target_code = sess.run(latent_w, feed_dict={real: target_image})

                codes = linear_interpolate(src_code=source_code,
                                           dst_code=target_code,
                                           step=args.step)

                inputs = np.zeros((args.batch_size, latent_dim), np.float32)
                output_images = []
                for idx in range(0, args.step, args.batch_size):
                    batch = codes[idx:idx + args.batch_size]
                    inputs[0:len(batch)] = batch
                    images = sess.run(reconstruction_from_w, feed_dict={Z: inputs})
                    output_images.append(images[0:len(batch)])
                output_images = np.concatenate(output_images, axis=0)
                final_results = np.concatenate([source_image, output_images, target_image], axis=0)
                final_results = final_results.transpose(0, 2, 3, 1)
                imwrite(immerge(final_results, 1, args.step + 2), '%s/%s_to_%s.png' %
                        (save_dir, source_name, target_name))
    else:
        raise ValueError('Invalid mode!')


if __name__ == "__main__":
    main()
