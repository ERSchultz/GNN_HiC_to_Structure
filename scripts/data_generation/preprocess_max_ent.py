import json
import math
import os
import os.path as osp
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import tqdm
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import (calculate_D, calculate_diag_chi_step,
                                      calculate_L, calculate_U)
from pylib.utils.load_utils import (get_final_max_ent_folder, load_L,
                                    load_max_ent_D, load_max_ent_L,
                                    load_max_ent_U, load_psi)
from pylib.utils.plotting_utils import plot_matrix, plot_seq_continuous
from pylib.utils.utils import LETTERS, load_json, triu_to_full
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from utils.utils import ROOT, get_samples


def calculate_chis_in_eigspace(dataset, b, phi, v, k, ar, cell_line=None):
    '''
    Calculate eigenvectors and eigenvalues of L matrix for use in data genration
    procedure. (Instead of fitting distributions to chi values, we will fit
    distributions to eigenvalues of L matrix - which can be thought of as projecting
    the chis in the the eigensapce of the L matrix).
    '''
    if cell_line is None:
        samples = get_samples(dataset, train=True)
    else:
        samples = get_samples(dataset, train=True, filter_cell_lines=[cell_line])

    print(f'{len(samples)} samples')
    for sample in samples:
        s_dir = osp.join(ROOT, dataset, f'samples/sample{sample}')
        print(sample)

        if v is None:
            max_ent_dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
        else:
            assert v is not None
            max_ent_dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
        if ar != 1:
            max_ent_dir += f'_spheroid_{ar}'
        max_ent_dir += f'-max_ent{k}'
        if not osp.exists(max_ent_dir):
            print(f'{max_ent_dir} does not exist')
            continue

        final = get_final_max_ent_folder(max_ent_dir)
        with open(osp.join(final, 'config.json')) as f:
            config = json.load(f)
            chis = np.array(config['chis'])
        plot_matrix(chis, osp.join(max_ent_dir, 'chis.png'), cmap = 'blue-red')

        # save x in resources
        x = load_psi(max_ent_dir)
        np.save(osp.join(max_ent_dir, 'resources/x.npy'), x)

        # compute EIG(L)
        L = calculate_L(x, chis)
        w, V = np.linalg.eig(L)
        x_eig = V[:,:k]

        assert np.sum((np.isreal(x_eig)))
        x_eig = np.real(x_eig)

        chis_eig = np.zeros_like(chis)
        for i, val in enumerate(w[:k]):
            assert np.isreal(val)
            chis_eig[i,i] = np.real(val)

        L_eig = x_eig @ chis_eig @ x_eig.T
        assert np.allclose(L, L_eig), L - L_eig

        # normalize x_eig
        x_eig_norm = np.zeros_like(x)
        chis_eig_norm = np.zeros_like(chis)
        for i in range(k):
            xi = x_eig[:, i]
            min = np.min(xi)
            max = np.max(xi)
            if max > abs(min):
                val = max
            else:
                val = abs(min)

            # multiply by scale such that val x scale = 1
            scale = 1/val
            x_eig_norm[:,i] = xi * scale

            # multiply by val**2 to counteract
            chis_eig_norm[i,i] = chis_eig[i,i] * val * val

        np.save(osp.join(max_ent_dir, 'resources/x_eig_norm.npy'), x_eig_norm)
        plot_seq_continuous(x_eig_norm,
                            ofile = osp.join(max_ent_dir, 'resources/x_eig_norm.png'))
        np.save(osp.join(max_ent_dir, 'chis_eig_norm.npy'), chis_eig_norm)
        plot_matrix(chis_eig_norm, osp.join(max_ent_dir, 'chis_eig_norm.png'),
                    cmap = 'blue-red')

def curve_fit_diag_chi(dataset, b, phi, v, k, ar, plot=True, cell_line=None):
    '''Fit polynomial curve to diagonal chis for use in data generation.'''
    if cell_line is None:
        samples = get_samples(dataset, train=True)
    else:
        samples = get_samples(dataset, train=True, filter_cell_lines=[cell_line])

    for sample in samples:
        print(f'sample {sample}')
        s_dir = osp.join(ROOT, dataset, f'samples/sample{sample}')
        if v is None:
            max_ent_dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
        else:
            assert v is not None
            max_ent_dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
        if ar != 1:
            max_ent_dir += f'_spheroid_{ar}'
        max_ent_dir += f'-max_ent{k}'
        if not osp.exists(max_ent_dir):
            print(f'{max_ent_dir} does not exist')
            continue
        odir = osp.join(max_ent_dir, 'fitting')
        os.makedirs(odir, exist_ok = True)

        final = get_final_max_ent_folder(max_ent_dir)
        ifile = osp.join(final, 'config.json')
        with open(ifile, 'r') as f:
            config = json.load(f)
        diag_chi_step = calculate_diag_chi_step(config)
        m = len(diag_chi_step)
        x = np.arange(0, 2*m)

        U = load_max_ent_U(max_ent_dir)
        meanDist_U = DiagonalPreprocessing.genomic_distance_statistics(U, 'freq')

        curves = [Curves.poly6_curve, Curves.poly8_curve, Curves.poly12_curve]
        colors = ['b', 'r', 'g']
        orders = [6, 8, 12]
        X = x[:m]
        if plot:
            plt.plot(X, meanDist_U, ls='-', c='k', label=r'$\delta^{ME(i)}$')
        for curve, color, order in zip(curves, colors, orders):
            init = [1]*(order+1)
            for start in [1, 2, 5]:
                poly_log_fit = curve_fit_helper(curve, np.log(x[:m]), meanDist_U,
                                                f'poly{order}_log_start_{start}_meanDist_U', odir,
                                                init, start = start)

                if plot:
                    plt.plot(X, poly_log_fit, ls=':', c=color,
                            label=r'$\hat{\delta}^{ME(i)}$' + f' (o={order}th, s={start})')

        if plot:
            plt.ylim(np.min(meanDist_U), np.max(meanDist_U))
            plt.xscale('log')
            plt.xlabel('d',fontsize=16)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(osp.join(odir, 'delta_vs_delta_hat.png'))
            plt.close()

def curve_fit_helper(fn, x, y, label, odir, init = [1,1], start = 0):
    try:
        popt, pcov = curve_fit(fn, x[start:], y[start:], p0 = init, maxfev = 2000)
        # print(f'\t{label} popt', popt)
        fit = fn(x[start:], *popt)
        if start > 0:
            fit = np.append(np.zeros(start), fit)
        np.savetxt(osp.join(odir, f'{label}_fit.txt'), fit)
        np.savetxt(osp.join(odir, f'{label}_popt.txt'), popt)
    except RuntimeError as e:
        fit = None
        print(f'{label}:', e)

    return fit

class Curves():
    def poly6_curve(x, A, B, C, D, E, F, G):
        result = A + B*x + C*x**2 + D*x**3 + E*x**4 + F*x**5 + G*x**6
        return result

    def poly8_curve(x, A, B, C, D, E, F, G, H, I):
        result = A + B*x + C*x**2 + D*x**3 + E*x**4 + F*x**5 + G*x**6  + H*x**7 + I*x**8
        return result

    def poly10_curve(x, A, B, C, D, E, F, G, H, I, J, K):
        result = A + B*x + C*x**2 + D*x**3 + E*x**4 + F*x**5 + G*x**6 + H*x**7
        result += I*x**8 + J*x**9 + K*x**10
        return result

    def poly12_curve(x, A, B, C, D, E, F, G, H, I, J, K, L, M):
        result = A + B*x + C*x**2 + D*x**3 + E*x**4 + F*x**5 + G*x**6 + H*x**7
        result += I*x**8 + J*x**9 + K*x**10 + L*x**11 + M*x**12
        return result

def simple_histogram(arr, xlabel='X', odir=None, ofname=None, dist=skewnorm,
                    label=None, legend_title='', color=None):
    title = []
    if arr is None:
        return
    arr = np.array(arr).reshape(-1)
    if color is None:
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                bins = 50, alpha = 0.5, label = label)
    else:
        n, bins, patches = plt.hist(arr, weights = np.ones_like(arr) / len(arr),
                                bins = 50, alpha = 0.5, label = label, color = color)
    bin_width = bins[1] - bins[0]
    if dist is not None:
        params = dist.fit(arr)
        y = dist.pdf(bins, *params) * bin_width
        params = [np.round(p, 3) for p in params]
        print(ofname, params)
        if dist == skewnorm and ofname is not None:
            with open(osp.join(odir, ofname[:-9]+'.pickle'), 'wb') as f:
                dict = {'alpha':params[0], 'mu':params[1], 'sigma':params[2]}
                pickle.dump(dict, f)
        plt.plot(bins, y, ls = '--', color = 'k')
    if not (odir is None or ofname is None):
        if label is not None:
            plt.legend(title = legend_title)
        plt.ylabel('probability', fontsize=16)
        plt.xlabel(xlabel, fontsize=16)
        # plt.xlim(-20, 20)
        if dist == skewnorm:
            title = r'$\alpha=$' + f'{params[0]}\n'
            title += r'$\mu$=' + f'{params[-2]} '+r'$\sigma$='+f'{params[-1]}'
            plt.title(title)
        elif dist == norm:
            title = r'$\mu$=' + f'{params[0]:.2f} '
            title += r'$\sigma$=' + f'{params[-1]:.2f}'
            plt.title(title)

        plt.savefig(osp.join(odir, ofname))
        plt.close()

def chis_eigspace_distribution(dataset, b, phi, v, k, ar, plot=True,
                cell_line=None,):
    '''
    Fit KDE distributions to eigenvalues of L matrix for use in data genration
    procedure. (Instead of fitting distributions to chi values, we will fit
    distributions to eigenvalues of L matrix - which can be thought of as projecting
    the chis in the the eigensapce of the L matrix).
    '''
    # distribution of plaid params
    if cell_line is None:
        samples, cell_lines = get_samples(dataset, True, return_cell_lines=True)
    elif isinstance(cell_line, str):
        samples, cell_lines = get_samples(dataset, True, return_cell_lines=True,
                                    filter_cell_lines=set([cell_line]))
    elif isinstance(cell_line, list):
        samples, cell_lines = get_samples(dataset, True, return_cell_lines=True,
                                    filter_cell_lines=set(cell_line))
    else:
        raise Exception(f'Cell lines unrecognized: {cell_line}')

    print(len(samples))
    N = len(samples)
    data_dir = osp.join(ROOT, dataset)

    if v is None:
        assert phi is not None
        odir = osp.join(data_dir, f'b_{b}_phi_{phi}')
    else:
        odir = osp.join(data_dir, f'b_{b}_v_{v}')
    if ar != 1.0:
        odir += f'_spheroid_{ar}'
    odir += '_distributions'
    if cell_line is not None:
        odir += f'_{cell_line}'
    os.makedirs(odir, exist_ok = True)


    odir = osp.join(odir, 'plaid_param_distributions_eig_norm')
    print(odir)
    os.makedirs(odir, exist_ok = True)

    L_list = []
    D_list = []
    U_list = []
    chi_ij_list = []
    chi_ii_list = []
    chi_list = []
    for sample in samples:
        s_dir = osp.join(data_dir, f'samples/sample{sample}')
        if v is None:
            s_dir = osp.join(s_dir, f'optimize_grid_b_{b}_phi_{phi}')
        else:
            assert v is not None
            s_dir = osp.join(s_dir, f'optimize_grid_b_{b}_v_{v}')
        if ar != 1:
            s_dir += f'_spheroid_{ar}'
        s_dir += f'-max_ent{k}'

        if not osp.exists(s_dir):
            print(f'WARNING: {s_dir} does not exist')
            continue

        # get L
        final = get_final_max_ent_folder(s_dir)
        L = load_max_ent_L(final, True)
        D = load_max_ent_D(s_dir)
        U = calculate_U(L, D)

        m = len(L)
        L_list.append(L[np.triu_indices(m)])
        D_list.append(D[np.triu_indices(m)])
        U_list.append(U[np.triu_indices(m)])

        # get chi
        chi = np.load(osp.join(s_dir, 'chis_eig_norm.npy'))

        if chi is not None:
            k = len(chi)
            chi_list.append(chi)
            for i in range(k):
                for j in range(i+1):
                    chi_ij = chi[i,j]
                    if i == j:
                        chi_ii_list.append(chi_ij)
                    else:
                        chi_ij_list.append(chi_ij)
        else:
            print('WARNING: chi is None')


    if plot:
        label_fontsize=24
        legend_fontsize=16
        tick_fontsize=22
        letter_fontsize=26
        # plot plaid chi parameters

        # plaid chi_ii parameters
        simple_histogram(chi_ii_list, r'$\chi_{ii}$', odir,
                            f'k{k}_chi_ii_dist.png', dist = skewnorm)

        # plaid per chi
        print("Starting plaid per chi")
        bin_width = 10
        cmap = matplotlib.cm.get_cmap('tab10')
        ind = np.arange(k) % cmap.N
        colors = cmap(ind.astype(int))
        # per chi ii
        rows = math.ceil(k / 5)
        cols = min(5, k)
        fig, ax = plt.subplots(rows, cols)
        fig.set_figheight(5)
        fig.set_figwidth(10)

        if rows == 1:
            ax = [ax]
        c = 0
        row = 0
        col = 0
        for i in range(k):
            print(f'k={i}')
            data = []
            for chi in chi_list:
                data.append(chi[i,i])
            arr = np.array(data).reshape(-1)

            # remove outliers by 1.5 * IQR
            iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
            width = 1.5
            l_cutoff = np.percentile(arr, 25) - width * iqr
            u_cutoff = np.percentile(arr, 75) + width * iqr
            delete_arr = np.logical_or(arr < l_cutoff, arr > u_cutoff)
            print(i, iqr, (l_cutoff, u_cutoff))
            arr = np.delete(arr, delete_arr, axis = None)
            #
            # remove outliers by zscore
            # mean = np.mean(arr)
            # std = np.std(arr)
            # delete_arr = np.abs(arr - mean)/ std > 2
            # print(np.sum(delete_arr))
            # arr = np.delete(arr, delete_arr, axis = None)

            dist = skewnorm

            bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
            n, bins, patches = ax[row][col].hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = bins, alpha = 0.5, color = colors[c])

            params = dist.fit(arr)
            y = dist.pdf(bins, *params) * bin_width
            params = np.round(params, 1)
            with open(osp.join(odir, f'k{k}_chi{LETTERS[i]}{LETTERS[i]}.pickle'), 'wb') as f:
                dict = {'alpha':params[0], 'mu':params[1], 'sigma':params[2]}
                pickle.dump(dict, f)


            ax[row][col].plot(bins, y, ls = '--', color = 'k')
            title = r'$\alpha=$' + f'{params[0]}\n'
            title += r'$\mu$=' + f'{params[-2]} '+r'$\sigma$='+f'{params[-1]}'
            # ax[row][col].set_title(title)
            ax[row][col].set_xlabel(rf'$\chi${LETTERS[i]+LETTERS[i]}')
            ax[row][col].set_xlabel(rf'$\lambda_{{{i+1}}}$', fontsize=16)
            # ax[row][col].tick_params(axis='both', which='major', labelsize=tick_fontsize)

            col += 1
            if col == cols:
                col = 0
                row += 1
            c += 1

        fig.supylabel('Probability', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'k{k}_chi_per_ii_dist.png'))
        plt.close()

        fig, ax = plt.subplots(rows, cols)
        fig.set_figheight(5)
        fig.set_figwidth(10)

        if rows == 1:
            ax = [ax]
        c = 0
        row = 0
        col = 0
        for i in range(k):
            print(f'k={i}')
            data = []
            for chi in chi_list:
                data.append(chi[i,i])
            arr = np.array(data).reshape(-1)

            # remove outliers by 1.5 * IQR
            iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
            width = 1.5
            l_cutoff = np.percentile(arr, 25) - width * iqr
            u_cutoff = np.percentile(arr, 75) + width * iqr
            delete_arr = np.logical_or(arr < l_cutoff, arr > u_cutoff)
            print(i, iqr, (l_cutoff, u_cutoff))
            arr = np.delete(arr, delete_arr, axis = None)

            # remove outliers by zscore
            # mean = np.mean(arr)
            # std = np.std(arr)
            # delete_arr = np.abs(arr - mean)/ std > 2
            # print(np.sum(delete_arr))
            # arr = np.delete(arr, delete_arr, axis = None)

            bins = range(math.floor(min(arr)), math.ceil(max(arr)) + bin_width, bin_width)
            n, bins, patches = ax[row][col].hist(arr, weights = np.ones_like(arr) / len(arr),
                                        bins = bins, alpha = 0.5, color = colors[c])
            ax2 = ax[row][col].twinx()

            kde = KernelDensity(kernel='gaussian', bandwidth=10)
            bins = np.array((range(math.floor(min(arr)), math.ceil(max(arr)) + 1, 1)))

            kde.fit(arr.reshape(-1, 1))
            y = kde.score_samples(bins.reshape(-1, 1)) * 1
            y = np.exp(y)

            with open(osp.join(odir, f'k{k}_chi{LETTERS[i]}{LETTERS[i]}_KDE.pickle'), 'wb') as f:
                pickle.dump(kde, f)


            ax2.plot(bins, y, ls = '--', color = 'k')
            ax2.set_yticks([])
            ax[row][col].set_yticks([])
            title = r'$\alpha=$' + f'{params[0]}\n'
            title += r'$\mu$=' + f'{params[-2]} '+r'$\sigma$='+f'{params[-1]}'
            # ax[row][col].set_title(title)
            ax[row][col].set_xlabel(rf'$\chi${LETTERS[i]+LETTERS[i]}')
            ax[row][col].set_xlabel(rf'$\lambda_{{{i+1}}}$', fontsize=16)
            # ax[row][col].tick_params(axis='both', which='major', labelsize=tick_fontsize)

            col += 1
            if col == cols:
                col = 0
                row += 1
            c += 1

        fig.supylabel('Probability', fontsize=16)
        plt.tight_layout()
        plt.savefig(osp.join(odir, f'k{k}_chi_per_ii_KDE.png'))
        plt.close()

        # meanDist_U
        for U in U_list:
            U = triu_to_full(U)
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(U, 'freq')
            plt.plot(meanDist)
        plt.xscale('log')
        plt.savefig(osp.join(odir, 'meanDist_U.png'))
        plt.close()

        # meanDist_U
        for D in D_list:
            D = triu_to_full(D)
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(D, 'freq')
            plt.plot(meanDist)
        plt.xscale('log')
        plt.savefig(osp.join(odir, 'meanDist_D.png'))
        plt.close()

    return L_list, U_list, D_list, chi_ij_list

def main():
    cell_line = ['gm12878', 'imr90', 'hap1', 'huvec', 'hmec']
    dataset = 'dataset_12_06_23'
    # dataset = 'dataset_all_files_50k_512'
    # calculate_chis_in_eigspace(dataset, b=200, phi=None, v=8, k=10, ar=1.5, cell_line=cell_line)
    # curve_fit_diag_chi(dataset, b=200, phi=None, v=8, k=10, ar=1.5,
    #                         plot=True, cell_line=cell_line)
    chis_eigspace_distribution(dataset, b=200, phi=None, v=8, k=10, ar=1.5, plot=True,
                cell_line=cell_line)

if __name__ == '__main__':
    main()
