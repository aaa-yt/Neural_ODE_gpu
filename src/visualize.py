from logging import getLogger
import matplotlib.pyplot as plt

from config import Config

logger = getLogger(__name__)

class Visualize:
    def __init__(self, config: Config):
        self.config = config
        self.fig_realtime = plt.figure(figsize=(10.,7.5))
        self.ax_loss = self.fig_realtime.add_subplot(231)
        if config.trainer.is_accuracy: self.ax_accuracy = self.fig_realtime.add_subplot(232)
        self.ax_params = self.fig_realtime.add_subplot(233)
        self.ax_alpha = self.fig_realtime.add_subplot(234)
        self.ax_beta = self.fig_realtime.add_subplot(235)
        self.ax_gamma = self.fig_realtime.add_subplot(236)
        self.cmap_alpha = plt.get_cmap("Reds")
        self.cmap_beta = plt.get_cmap("Blues")
        self.cmap_gamma = plt.get_cmap("Greens")
        self.label = ['Train', 'Validation']
    
    def plot_realtime(self, t, params, losses, accuracies=None):
        alpha, beta, gamma = params
        epoch = [i for i in range(1, len(losses[0])+1)]
        self.ax_loss.cla()
        for i, loss in enumerate(losses):
            self.ax_loss.plot(epoch, loss, label=self.label[i])
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Loss')
        self.ax_loss.legend()
        if accuracies is not None:
            self.ax_accuracy.cla()
            for i, accuracy in enumerate(accuracies):
                self.ax_accuracy.plot(epoch, accuracy, label=self.label[i])
            self.ax_accuracy.set_xlabel('Epoch')
            self.ax_accuracy.set_ylabel('Accuracy')
            self.ax_accuracy.set_title('Accuracy')
            self.ax_accuracy.legend()
        self.ax_params.cla()
        self.ax_alpha.cla()
        for i in range(len(alpha[0])):
            nc = (float(i) / (2 * len(alpha[0]))) + 0.5
            self.ax_alpha.plot(t, alpha[:,i], label=r'$\alpha_{}(t)$'.format(i), c=self.cmap_alpha(nc))
            self.ax_params.plot(t, alpha[:,i], label=r'$\alpha_{}(t)$'.format(i), c=self.cmap_alpha(nc))
        self.ax_alpha.set_xlabel('t')
        self.ax_alpha.set_ylabel(r'$\alpha(t)$')
        self.ax_alpha.set_title(r'$\alpha(t)$')
        #self.ax_alpha.legend()
        self.ax_beta.cla()
        for i in range(len(beta[0])):
            for j in range(len(beta[0,0])):
                nc = (float(i * len(beta[0,0]) + j) / (2 * len(beta[0]) * len(beta[0,0]))) + 0.5
                self.ax_beta.plot(t, beta[:, i, j], label=r'$\beta_{}$$_{}(t)$'.format(i, j), c=self.cmap_beta(nc))
                self.ax_params.plot(t, beta[:, i, j], label=r'$\beta_{}$$_{}(t)$'.format(i, j), c=self.cmap_beta(nc))
        self.ax_beta.set_xlabel('t')
        self.ax_beta.set_ylabel(r'$\beta(t)$')
        self.ax_beta.set_title(r'$\beta(t)$')
        #self.ax_beta.legend()
        self.ax_gamma.cla()
        for i in range(len(gamma[0])):
            nc = (float(i) / (2 * len(gamma[0]))) + 0.5
            self.ax_gamma.plot(t, gamma[:,i], label=r'$\gamma_{}(t)$'.format(i), c=self.cmap_gamma(nc))
            self.ax_params.plot(t, gamma[:,i], label=r'$\gamma_{}(t)$'.format(i), c=self.cmap_gamma(nc))
        self.ax_gamma.set_xlabel('t')
        self.ax_gamma.set_ylabel(r'$\gamma(t)$')
        self.ax_gamma.set_title(r'$\gamma(t)$')
        #self.ax_gamma.legend()
        self.ax_params.set_xlabel('t')
        self.ax_params.set_title('Parameter')
        #self.ax_params.legend()
        self.fig_realtime.tight_layout()
        self.fig_realtime.suptitle('Epoch: {}'.format(epoch[-1]))
        self.fig_realtime.subplots_adjust(top=0.92)
        plt.draw()
        plt.pause(0.0000000001)
    
    def save_plot_loss(self, losses, xlabel=None, ylabel=None, title=None, save_file=None):
        plt.clf()
        epoch = [i for i in range(len(losses[0]))]
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if title is not None: plt.title(title)
        for i, loss in enumerate(losses):
            plt.plot(epoch, loss, label=self.label[i])
        plt.legend()
        if save_file is None:
            plt.show()
        else:
            logger.debug("save plot of loss to {}".format(save_file))
            plt.savefig(save_file)
    
    def save_plot_accuracy(self, accuracies, xlabel=None, ylabel=None, title=None, save_file=None):
        plt.cla()
        if not self.config.trainer.is_accuracy: return
        epoch = [i for i in range(len(accuracies[0]))]
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if title is not None: plt.title(title)
        for i, accuracy in enumerate(accuracies):
            plt.plot(epoch, accuracy, label=self.label[i])
        plt.legend()
        if save_file is None:
            plt.show()
        else:
            logger.debug("save plot of accuracy to {}".format(save_file))
            plt.savefig(save_file)
    
    def save_plot_params(self, t, params, save_file=None):
        plt.cla()
        alpha, beta, gamma = params
        fig_alpha = plt.figure()
        ax_alpha = fig_alpha.add_subplot(111)
        ax_alpha.set_xlabel("t")
        ax_alpha.set_ylabel(r'$\alpha(t)$')
        fig_beta = plt.figure()
        ax_beta = fig_beta.add_subplot(111)
        ax_beta.set_xlabel("t")
        ax_beta.set_ylabel(r'$\beta(t)$')
        fig_gamma = plt.figure()
        ax_gamma = fig_gamma.add_subplot(111)
        ax_gamma.set_xlabel("t")
        ax_gamma.set_ylabel(r'$\gamma(t)$')
        fig_params = plt.figure()
        ax_params = fig_params.add_subplot(111)
        ax_params.set_xlabel("t")
        
        for i in range(len(alpha[0])):
            ax_alpha.plot(t, alpha[:,i], label=r'$\alpha_{}(t)$'.format(i))
            ax_params.plot(t, alpha[:,i], label=r'$\alpha_{}(t)$'.format(i))
        for i in range(len(beta[0])):
            for j in range(len(beta[0,0])):
                ax_beta.plot(t, beta[:, i, j], label=r'$\beta_{}$$_{}(t)$'.format(i, j))
                ax_params.plot(t, beta[:, i, j], label=r'$\beta_{}$$_{}(t)$'.format(i, j))
        for i in range(len(gamma[0])):
            ax_gamma.plot(t, gamma[:,i], label=r'$\gamma_{}(t)$'.format(i))
            ax_params.plot(t, gamma[:,i], label=r'$\gamma_{}(t)$'.format(i))
        
        ax_alpha.legend()
        ax_beta.legend()
        ax_gamma.legend()
        ax_params.legend()
        if save_file is not None:
            logger.debug("save plot of parameter0 (alpha) to {}".format(save_file[0]))
            fig_alpha.savefig(save_file[0])
            logger.debug("save plot of parameter1 (beta) to {}".format(save_file[1]))
            fig_beta.savefig(save_file[1])
            logger.debug("save plot of parameter2 (gamma) to {}".format(save_file[2]))
            fig_gamma.savefig(save_file[2])
            logger.debug("save plot of parameters to {}".format(save_file[3]))
            fig_params.savefig(save_file[3])