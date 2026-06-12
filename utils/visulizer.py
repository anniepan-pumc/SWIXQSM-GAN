import matplotlib.pyplot as plt
import os


def draw_curve(gen_loss, disc_loss, fids_t2f, fids_q2s, current_epoch, save_path = "./save"):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="fid")
    ax0.plot(len(gen_loss), gen_loss, 'bo-', label='gen')
    ax0.plot(len(disc_loss), disc_loss, 'ro-', label='disc')
    ax1.plot(len(fids_t2f), fids_t2f, 'bo-', label='t2f')
    ax1.plot(len(fids_q2s), fids_q2s, 'ro-', label='q2s')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join(save_path, 'result.jpg'))
