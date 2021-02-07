from __future__ import print_function, unicode_literals
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from .fh_utils import *

def show_training_samples(base_path, version, num2show=None, render_mano=False):
    if render_mano:
        from frei_hand.utils import HandModel, recover_root, get_focal_pp, split_theta

    if num2show == -1:
        num2show = db_size('training') # show all
    file_path = os.path.realpath(__file__) # current location
    dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_path))) # find dataset location

    base_path = os.path.join(dataset_dir, base_path)
    print(base_path)
    # load annotations
    K_list, mano_list, xyz_list = load_db_annotation(base_path, 'training')

    # iterate over all samples
    for idx in range(117216, 117216+1):
        idx %= 32560

        # load image and mask
        img = read_img(idx, base_path, 'training', version)
        print(img.shape)
        # msk = read_msk(idx, base_path)

        # annotation for this frame
        K, mano, xyz = K_list[idx],mano_list[idx], xyz_list[idx]
        # print(db_data_anno)
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)
        print('The notations of the', idx, ' sample')
        print(uv)
        # render an image of the shape
        msk_rendered = None
        if render_mano:
            # split mano parameters
            poses, shapes, uv_root, scale = split_theta(mano)
            focal, pp = get_focal_pp(K)
            xyz_root = recover_root(uv_root, scale, focal, pp)

            # set up the hand model and feed hand parameters
            renderer = HandModel(use_mean_pca=False, use_mean_pose=True)
            renderer.pose_by_root(xyz_root[0], poses[0], shapes[0])
            msk_rendered = renderer.render(K, img_shape=img.shape[:2])

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img)
        # ax2.imshow(msk if msk_rendered is None else msk_rendered)
        plot_hand(ax1, uv, order='uv')
        # plot_hand(ax2, uv, order='uv')
        ax1.axis('off')
        ax2.axis('off')
        file_path = os.path.realpath(__file__)  # current location
        dataset_dir = os.path.dirname(file_path)
        plt.savefig(os.path.join(dataset_dir, 'showed_samples/notated'+str(idx)+'.png'))
        plt.show()

def show_result(img, prediction):
    print('The notations of the sample')
    print(prediction)
    # show
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(img)
    # ax2.imshow(msk if msk_rendered is None else msk_rendered)
    plot_hand(ax1, prediction, order='uv')
    # plot_hand(ax2, uv, order='uv')
    ax1.axis('off')
    ax2.axis('off')
    file_path = os.path.realpath(__file__)  # current location
    dataset_dir = os.path.dirname(file_path)
    plt.savefig(os.path.join(dataset_dir, 'showed_samples/Predicted' + str(idx) + '.png'))
    plt.show()


def show_eval_samples(base_path, num2show=None):
    if num2show == -1:
        num2show = db_size('evaluation') # show all

    for idx in  range(db_size('evaluation')):
        if idx >= num2show:
            break

        # load image only, because for the evaluation set there is no mask
        img = read_img(idx, base_path, 'evaluation')

        # show
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)
        ax1.axis('off')
        plt.show()


if __name__ == '__main__':
    # >> sudo python view_samples.py FreiHAND_pub_v2
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str, default= 'FreiHAND_pub_v2',
                        help='Path to where the FreiHAND dataset is located.')
    parser.add_argument('--show_eval', action='store_true',
                        help='Shows samples from the evaluation split if flag is set, shows training split otherwise.')
    parser.add_argument('--mano', action='store_true',
                        help='Enables rendering of the hand if mano is available. See README for details.')
    parser.add_argument('--num2show', type=int, default=10,
                        help='Number of samples to show. ''-1'' defaults to show all.')
    parser.add_argument('--sample_version', type=str, default=sample_version.gs,
                        help='Which sample version to use when showing the training set.'
                             ' Valid choices are %s' % sample_version.valid_options())
    args = parser.parse_args()

    # check inputs
    msg = 'Invalid choice: ''%s''. Must be in %s' % (args.sample_version, sample_version.valid_options())
    assert args.sample_version in sample_version.valid_options(), msg

    if args.show_eval:
        """ Show some evaluation samples. """
        show_eval_samples(args.base_path,
                          num2show=args.num2show)

    else:
        """ Show some training samples. """
        show_training_samples(
            args.base_path,
            args.sample_version,
            num2show=args.num2show,
            render_mano=None
        )

