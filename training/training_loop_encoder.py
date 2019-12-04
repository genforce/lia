import tensorflow as tf
import numpy as np
import os
import sys
from utils import imwrite, immerge
import dnnlib
from dnnlib import EasyDict
import dnnlib.tflib as tflib
from training import misc
from perceptual_model import PerceptualModel
import time

def process_reals(x, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        return x


def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])


def get_train_data(sess, data_dir, submit_config, mode):
    if mode == 'train':
        shuffle = True; repeat = True; batch_size = submit_config.batch_size
    elif mode == 'test':
        shuffle = False; repeat = True; batch_size = submit_config.batch_size_test
    else:
        raise Exception("mode must in ['train', 'test], but got {}" % mode)

    dset = tf.data.TFRecordDataset(data_dir)
    dset = dset.map(parse_tfrecord_tf, num_parallel_calls=16)
    if shuffle:
        bytes_per_item = np.prod([3, submit_config.image_size, submit_config.image_size]) * np.dtype('uint8').itemsize
        dset = dset.shuffle(((4096 << 20) - 1) // bytes_per_item + 1)
    if repeat:
        dset = dset.repeat()
    dset = dset.batch(batch_size)
    train_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
    training_init_op = train_iterator.make_initializer(dset)
    image_batch = train_iterator.get_next()
    sess.run(training_init_op)
    return image_batch


def test(E, Gs, real_test, submit_config):
    with tf.name_scope("Run"), tf.control_dependencies(None):
        with tf.device("/cpu:0"):
            in_split = tf.split(real_test, submit_config.num_gpus)
        out_split = []
        num_layers = Gs.components.synthesis.input_shape[1]
        for gpu in range(submit_config.num_gpus):
            with tf.device("/gpu:%d" % gpu):
                in_gpu = in_split[gpu]
                latent_w_val = E.get_output_for(in_gpu, phase=False)
                latent_w_val = tf.tile(latent_w_val[:, np.newaxis], [1, num_layers, 1])
                fake_X_val = Gs.components.synthesis.get_output_for(latent_w_val, randomize_noise=False)
                out_split.append(fake_X_val)

        with tf.device("/cpu:0"):
            out_expr = tf.concat(out_split, axis=0)

    return out_expr

def training_loop(
                  submit_config,
                  Encoder_args            = {},
                  E_opt_args              = {},
                  D_opt_args              = {},
                  E_loss_args             = {},
                  D_loss_args             = {},
                  lr_args                 = EasyDict(),
                  tf_config               = {},
                  dataset_args            = EasyDict(),
                  decoder_pkl             = EasyDict(),
                  drange_data             = [0, 255],
                  drange_net              = [-1,1],    # Dynamic range used when feeding image data to the networks.
                  mirror_augment          = False,
                  resume_run_id           = None,      # Run ID or network pkl to resume training from, None = start from scratch.
                  resume_snapshot         = None,      # Snapshot index to resume training from, None = autodetect.
                  image_snapshot_ticks    = 1,         # How often to export image snapshots?
                  network_snapshot_ticks  = 10,        # How often to export network snapshots?
                  save_tf_graph=False,                 # Include full TensorFlow computation graph in the tfevents file?
                  save_weight_histograms=False,        # Include weight histograms in the tfevents file?
                  max_iters               = 150000,
                  E_smoothing             = 0.999):

    tflib.init_tf(tf_config)

    with tf.name_scope('input'):
        real_train = tf.placeholder(tf.float32, [submit_config.batch_size, 3, submit_config.image_size, submit_config.image_size], name='real_image_train')
        real_test = tf.placeholder(tf.float32, [submit_config.batch_size_test, 3, submit_config.image_size, submit_config.image_size], name='real_image_test')
        real_split = tf.split(real_train, num_or_size_splits=submit_config.num_gpus, axis=0)

    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            E, G, D, Gs, NE = misc.load_pkl(network_pkl)
            start = int(network_pkl.split('-')[-1].split('.')[0]) // submit_config.batch_size
        else:
            print('Constructing networks...')
            G, D, Gs, NE = misc.load_pkl(decoder_pkl.decoder_pkl)
            E = tflib.Network('E', size=submit_config.image_size, filter=64, filter_max=1024, phase=True, **Encoder_args)
            start = 0

    Gs.print_layers(); E.print_layers(); D.print_layers()

    global_step = tf.Variable(start, trainable=False, name='learning_rate_step')
    learning_rate = tf.train.exponential_decay(lr_args.learning_rate, global_step, lr_args.decay_step,
                                               lr_args.decay_rate, staircase=lr_args.stair)
    add_global = global_step.assign_add(1)
    E_opt = tflib.Optimizer(name='TrainE', learning_rate=learning_rate, **E_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', learning_rate=learning_rate, **D_opt_args)

    E_loss_rec = 0.
    E_loss_adv = 0.
    D_loss_real = 0.
    D_loss_fake = 0.
    D_loss_grad = 0.
    for gpu in range(submit_config.num_gpus):
        print('build graph on gpu %s' % str(gpu))
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            E_gpu = E if gpu == 0 else E.clone(E.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')
            G_gpu = Gs if gpu == 0 else Gs.clone(Gs.name + '_shadow')
            perceptual_model = PerceptualModel(img_size=[submit_config.image_size, submit_config.image_size], multi_layers=False)
            real_gpu = process_reals(real_split[gpu], mirror_augment, drange_data, drange_net)
            with tf.name_scope('E_loss'), tf.control_dependencies(None):
                E_loss, recon_loss, adv_loss = dnnlib.util.call_func_by_name(E=E_gpu, G=G_gpu, D=D_gpu, perceptual_model=perceptual_model, reals=real_gpu, **E_loss_args)
                E_loss_rec += recon_loss
                E_loss_adv += adv_loss
            with tf.name_scope('D_loss'), tf.control_dependencies(None):
                D_loss, loss_fake, loss_real, loss_gp = dnnlib.util.call_func_by_name(E=E_gpu, G=G_gpu, D=D_gpu, reals=real_gpu, **D_loss_args)
                D_loss_real += loss_real
                D_loss_fake += loss_fake
                D_loss_grad += loss_gp
            with tf.control_dependencies([add_global]):
                E_opt.register_gradients(E_loss, E_gpu.trainables)
                D_opt.register_gradients(D_loss, D_gpu.trainables)

    E_loss_rec /= submit_config.num_gpus
    E_loss_adv /= submit_config.num_gpus
    D_loss_real /=submit_config.num_gpus
    D_loss_fake /=submit_config.num_gpus
    D_loss_grad /=submit_config.num_gpus

    E_train_op = E_opt.apply_updates()
    D_train_op = D_opt.apply_updates()

    #Es_update_op = Es.setup_as_moving_average_of(E, beta=E_smoothing)

    print('building testing graph...')
    fake_X_val = test(E, Gs, real_test, submit_config)

    sess = tf.get_default_session()

    print('Getting training data...')
    image_batch_train = get_train_data(sess, data_dir=dataset_args.data_train, submit_config=submit_config, mode='train')
    image_batch_test = get_train_data(sess, data_dir=dataset_args.data_test, submit_config=submit_config, mode='test')

    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        E.setup_weight_histograms(); D.setup_weight_histograms()

    cur_nimg = start * submit_config.batch_size
    cur_tick = 0
    tick_start_nimg = cur_nimg
    start_time = time.time()

    print('Optimization starts!!!')
    for it in range(start, max_iters):

        feed_dict = {real_train: sess.run(image_batch_train)}
        sess.run([E_train_op, E_loss_rec, E_loss_adv], feed_dict)
        sess.run([D_train_op, D_loss_real, D_loss_fake, D_loss_grad], feed_dict)

        cur_nimg += submit_config.batch_size

        if it % 100 == 0:
            print("Iter: %06d  kimg: %-8.1f time: %-12s" % (it, cur_nimg/1000, dnnlib.util.format_time(time.time()-start_time)))
            sys.stdout.flush()
            tflib.autosummary.save_summaries(summary_log, it)

        if cur_nimg >= tick_start_nimg + 65000:
            cur_tick += 1
            tick_start_nimg = cur_nimg

            if cur_tick % image_snapshot_ticks == 0:
                batch_images_test = sess.run(image_batch_test)
                batch_images_test = misc.adjust_dynamic_range(batch_images_test.astype(np.float32), [0, 255], [-1., 1.])

                samples2 = sess.run(fake_X_val, feed_dict={real_test: batch_images_test})
                samples2 = samples2.transpose(0, 2, 3, 1)
                batch_images_test = batch_images_test.transpose(0, 2, 3, 1)
                orin_recon = np.concatenate([batch_images_test, samples2], axis=0)
                imwrite(immerge(orin_recon, 2, submit_config.batch_size_test), '%s/iter_%08d.png' % (submit_config.run_dir, cur_nimg))

            if cur_tick % network_snapshot_ticks == 0:
                pkl = os.path.join(submit_config.run_dir, 'network-snapshot-%08d.pkl' % (cur_nimg))
                misc.save_pkl((E, G, D, Gs, NE), pkl)

    misc.save_pkl((E, G, D, Gs, NE), os.path.join(submit_config.run_dir, 'network-final.pkl'))
    summary_log.close()
