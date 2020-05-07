import torch as tr


class Hyperparams:
    """
    Base Hyperparams class.
    It uses base version of dcgan with 1D x space and z space
    """

    dtype = float
    n_iterations = 10000

    show_visual_while_training = True
    train_generator_adv = True
    train_autoencoder = True

    train_batch_logits = True
    train_sample_logits = True

    start_tensorboard = True

    train_generator = False
    train_discriminator = True

    gen_iter_count = 40
    disc_iter_count = 80
    step_ratio = gen_iter_count, disc_iter_count

    batch_size = 64
    seed_batch_size = 2048

    logit_x_batch_size = 16
    logit_z_batch_size = 16

    input_size = 2
    z_size = 2

    # Distribution params
    z_bounds = 4.0
    cor = 0.0

    # Learning Parameters
    lr_autoencoder = 0.0001
    lr_decoder = 0.0001
    lr_disc = 0.0001

    z_dist_type = 'normal'  # ['uniform', 'normal', 'sphere']

    model = 'bcgan'
    exp_name = 'trial_with_gmms'

    # dataloader = 'four_gaussian_sym'
    dataloader = 'nine_gaussian'

    child_iter = 50

    input_channel = 1
    input_height = 28
    input_width = 28

    @classmethod
    def z_means(cls):
        return tr.zeros(cls.z_size)

    @classmethod
    def z_cov(cls, sign='0'):
        cov = tr.eye(cls.z_size)
        # cor = {
        #     '+': cls.cor,
        #     '-': -cls.cor,
        #     '0': 0.0
        # }[sign]
        # cov[0, 1] = cov[1, 0] = cor
        return cov
