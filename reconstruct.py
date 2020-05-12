"""
Reconstruction given images using LIA, the reconstruction result will be saved.
"""
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
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified, '
                             '`./outputs/reconstruction` will be used by default.')
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
    E, _, _, Gs, _ = load_pkl(args.restore_path)
    num_layers = Gs.components.synthesis.input_shape[1]

    # Building graph
    real = tf.placeholder('float32', [None, 3, args.image_size, args.image_size], name='real_image')
    encoder_w = E.get_output_for(real, phase=False)
    encoder_w_tile = tf.tile(encoder_w[:, np.newaxis], [1, num_layers, 1])
    reconstructor = Gs.components.synthesis.get_output_for(encoder_w_tile, randomize_noise=False)
    sess = tf.get_default_session()

    # Preparing data
    input_images, _ = preparing_data(im_path=args.data_dir_test, img_type=args.img_type)

    save_dir = args.output_dir or './outputs/reconstruction'
    os.makedirs(save_dir, exist_ok=True)

    print('Reconstructing...')
    for it, image_id in tqdm(enumerate(range(0, input_images.shape[0], args.batch_size))):
        batch_images = input_images[image_id:image_id+args.batch_size]
        rec = sess.run(reconstructor, feed_dict={real: batch_images})
        orin_recon = np.concatenate([batch_images, rec], axis=0)
        orin_recon = orin_recon.transpose(0, 2, 3, 1)
        imwrite(immerge(orin_recon, 2, batch_images.shape[0]),
                '%s/reconstruction_%06d.png' % (save_dir, it))


if __name__ == "__main__":
    main()
