import tensorflow as tf
import numpy as np
import os
from utils import imwrite, immerge
import sys
import pickle
import dnnlib
import dnnlib.tflib as tflib
from tqdm import tqdm
import argparse
import glob
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(file_or_url):
    with open(file_or_url, 'rb') as file:
        return pickle.load(file, encoding='latin1')


def os_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sample(sess, Z, generator, save_dir):
    for it in tqdm(range(3)):
        samples1 = sess.run(generator, feed_dict={Z: np.random.randn(args.batch_size*2, args.z_dim)})
        samples1 = samples1.transpose(0, 2, 3, 1)
        samples1 = np.clip(samples1, -1., 1.)
        imwrite(immerge(samples1[:16, :, :, :], 2, 8), '%s/iter_%06d_sampling.png' % (save_dir, it))


def reconstruction(sess, images, real, reconstructor, save_dir):
    it = 0
    for image_id in tqdm(range(0, len(images), args.batch_size)):
        images_name = images[image_id:image_id+args.batch_size]
        batch_images = []
        for im_name in images_name:
            batch_images.append(cv2.imread(im_name)[:, :, ::-1])
        batch_images = np.asarray(batch_images)
        batch_images = adjust_dynamic_range(batch_images.astype(np.float32), [0, 255], [-1., 1.])
        rec = sess.run(reconstructor, feed_dict={real: batch_images.transpose(0, 3, 1, 2)})
        rec = rec.transpose(0, 2, 3, 1)
        rec = np.clip(rec, -1., 1.)
        orin_recon = np.concatenate([batch_images, rec], axis=0)
        imwrite(immerge(orin_recon, 2, len(images_name)), '%s/iter_%06d.png' % (save_dir, it))
        it += 1


def interpolation_on_z(sess, images, real, Z, get_z_from_img, get_img_from_z, step, save_dir):
    for image1 in tqdm(images):
        img_1 = cv2.imread(image1)[:, :, ::-1][np.newaxis]
        img_1 = adjust_dynamic_range(img_1.astype(np.float32), [0, 255], [-1., 1.])
        img_1_name = image1.split('/')[-1].split('.')[0]
        for image2 in images:

            img_2 = cv2.imread(image2)[:, :, ::-1][np.newaxis]
            img_2 = adjust_dynamic_range(img_2.astype(np.float32), [0, 255], [-1., 1.])
            img_2_name = image2.split('/')[-1].split('.')[0]

            latent_1 = sess.run(get_z_from_img, feed_dict={real: img_1.transpose(0, 3, 1, 2)})
            latent_2 = sess.run(get_z_from_img, feed_dict={real: img_2.transpose(0, 3, 1, 2)})

            linspace = np.linspace(0.0, 1.0, step)[:, np.newaxis].astype(np.float32)
            mid_res = latent_1 + linspace * (latent_2 - latent_1)

            mid_res = sess.run(get_img_from_z, feed_dict={Z: mid_res})
            mid_res = mid_res.transpose(0, 2, 3, 1)
            mid_res = np.clip(mid_res, -1., 1.)
            mid_res = np.concatenate([img_1, mid_res, img_2], axis=0)

            imwrite(immerge(mid_res, 1, step+2), '%s/%s_%s.png' % (save_dir, img_1_name, img_2_name))


def interpolation_on_w(sess, images, real, Z, get_w_from_img, get_img_from_w, step, save_dir):
    for image1 in tqdm(images):
        img_1 = cv2.imread(image1)[:, :, ::-1][np.newaxis]
        img_1 = adjust_dynamic_range(img_1.astype(np.float32), [0, 255], [-1., 1.])
        img_1_name = image1.split('/')[-1].split('.')[0]
        for image2 in images:
            img_2 = cv2.imread(image2)[:, :, ::-1][np.newaxis]
            img_2 = adjust_dynamic_range(img_2.astype(np.float32), [0, 255], [-1., 1.])
            img_2_name = image2.split('/')[-1].split('.')[0]

            latent_1 = sess.run(get_w_from_img, feed_dict={real: img_1.transpose(0, 3, 1, 2)})
            latent_2 = sess.run(get_w_from_img, feed_dict={real: img_2.transpose(0, 3, 1, 2)})

            linspace = np.linspace(0.0, 1.0, step)[:, np.newaxis].astype(np.float32)
            mid_res = latent_1 + linspace * (latent_2 - latent_1)

            mid_res = sess.run(get_img_from_w, feed_dict={Z: mid_res})
            mid_res = mid_res.transpose(0, 2, 3, 1)
            mid_res = np.clip(mid_res, -1., 1.)
            mid_res = np.concatenate([img_1, mid_res, img_2], axis=0)

            imwrite(immerge(mid_res, 1, step + 2), '%s/%s_%s.png' % (save_dir, img_1_name, img_2_name))


def manipulation(sess, images, real, Z, get_w_from_img, get_img_from_w, boundaries, save_dir, steps=11, start_distance=-5., end_distance=5.):
    linspace = np.linspace(start_distance, end_distance, steps)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    for boundary in boundaries:
        attr = boundary.split('/')[-1].split('_')[1]
        print('manipulating on %s' % (attr))
        boundary_ = np.load(boundary)
        boundary_ = boundary_ * linspace
        for image in tqdm(images):
            img_1 = cv2.imread(image)[:, :, ::-1][np.newaxis]
            img_1_name = image.split('/')[-1].split('.')[0]
            img_1 = adjust_dynamic_range(img_1.astype(np.float32), [0, 255], [-1., 1.])
            latent_w = sess.run(get_w_from_img, feed_dict={real: img_1.transpose(0, 3, 1, 2)})
            inter = latent_w + boundary_
            mid_res = sess.run(get_img_from_w, feed_dict={Z: inter})
            mid_res = mid_res.transpose(0, 2, 3, 1)
            mid_res = np.concatenate([img_1, mid_res], axis=0)
            imwrite(immerge(mid_res, 1, steps + 1), '%s/%s_attr_%s.png' % (save_dir, img_1_name, attr))



def main():

    tf_config = {'rnd.np_random_seed': 1000}
    tflib.init_tf(tf_config)
    E, G, D, Gs, NE = load_pkl(args.restore_path)
    num_layers = Gs.components.synthesis.input_shape[1]

    real = tf.placeholder('float32', [None, 3, args.image_size, args.image_size], name='real_image')
    Z = tf.placeholder('float32', [None, args.z_dim], name='Gaussian')

    sess = tf.get_default_session()

    # For sampling
    sampling_from_z = Gs.get_output_for(Z, None, randomize_noise=True)

    # For reconstruction
    encoder_w = E.get_output_for(real, phase=False)
    encoder_w_tile = tf.tile(encoder_w[:, np.newaxis], [1, num_layers, 1])
    reconstructor = Gs.components.synthesis.get_output_for(encoder_w_tile, randomize_noise=False)

    # For interpolation
    # interpolation on z
    N_Z = NE.get_output_for(encoder_w, None)
    reconstruction_from_z = Gs.get_output_for(Z, None, randomize_noise=False, is_validation=True)

    # interpolation on w
    W = tf.tile(Z[:, np.newaxis], [1, num_layers, 1])
    reconstruction_from_w = Gs.components.synthesis.get_output_for(W, randomize_noise=False)

    # Using ND's weights to setting NE's weights (invertible)
    ND_vars_pairs = {name: tflib.run(val) for name, val in Gs.components.mapping.vars.items()}
    for ne_name, ne_val in NE.vars.items():
        tflib.set_vars({ne_val: ND_vars_pairs[ne_name]})


    if args.mode == 0:
        save_dir = './test/sampling'
        os_mkdirs(save_dir)
        print('Sampling...')
        sample(sess, Z, generator=sampling_from_z, save_dir=save_dir)
    elif args.mode == 1:
        images = sorted(glob.glob(args.data_dir_test + '/*.png'))
        save_dir = './test/reconstruction'
        os_mkdirs(save_dir)
        print('Reconstructing...')
        reconstruction(sess, images, real, reconstructor=reconstructor, save_dir=save_dir)
    elif args.mode == 2:
        images = sorted(glob.glob(args.data_dir_test + '/*.png'))
        save_dir = './test/interpolation_on_z'
        os_mkdirs(save_dir)
        print('Interpolation in z space...')
        interpolation_on_z(sess, images, real, Z, get_z_from_img=N_Z, get_img_from_z=reconstruction_from_z, step=args.step, save_dir=save_dir)
    elif args.mode == 3:
        images = sorted(glob.glob(args.data_dir_test + '/*.png'))
        save_dir = './test/interpolation_on_w'
        os_mkdirs(save_dir)
        print('Interpolation in w space...')
        interpolation_on_w(sess, images, real, Z, get_w_from_img=encoder_w, get_img_from_w=reconstruction_from_w, step=args.step, save_dir=save_dir)
    elif args.mode == 4:
        images = sorted(glob.glob(args.data_dir_test + '/*.png'))
        boundaries = sorted(glob.glob(args.boundaries + '/*.npy'))
        save_dir = './test/manipulation'
        os_mkdirs(save_dir)
        print('manipulation in w space...')
        manipulation(sess, images, real, Z, get_w_from_img=encoder_w, get_img_from_w=reconstruction_from_w, boundaries=boundaries, save_dir=save_dir)
    else:
        raise ValueError('Invalid mode!')


if __name__ == "__main__":

    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir_test", type=str, default='',
                        help="Location of the test data")
    parser.add_argument("--boundaries", type=str, default='',
                        help="Location of the boundaries file")
    parser.add_argument('--restore_path', type=str, default='./network-final-ffhq.pkl',
                        help='The pre-trained encoder pkl file path')
    parser.add_argument("--image_size", type=int,
                        default=128, help="the training image size")
    parser.add_argument("--batch_size", type=int,
                        default=8, help="size of the input batch")
    parser.add_argument("--z_dim", type=int,
                        default=512, help="the dimension of the latent size")
    parser.add_argument("--step", type=int,
                        default=8, help="interpolation steps between two images")
    parser.add_argument("--mode", type=int,
                        default=0, help="to do one specified task")

    args = parser.parse_args()
    main()


