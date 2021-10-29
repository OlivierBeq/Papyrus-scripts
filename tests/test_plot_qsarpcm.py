# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def show_values_on_bars(axs, h_v="v", space=0.01):
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = '{:.2f}'.format(p.get_height())
                if p.get_height() < 0:
                    if _y < -3.5:
                        _y = 0.1
                        ax.text(_x, _y, value, ha="center", fontsize='small', rotation='vertical')
                    else:
                        ax.text(_x, _y, value, ha="center", va='top', fontsize='small', rotation='vertical')
                else:
                    _y += space
                    ax.text(_x, _y, value, ha="center", fontsize='small', rotation='vertical')


        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                if p.get_width() < 0:
                    _x -= space
                else:
                    _x += space
                value = '{:.2f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="left", fontsize='xx-small')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


if __name__ == '__main__':
    figsize = (8, 14)
    dpi = 600
    folder = os.path.dirname(__file__)
    sns.set_style("whitegrid")
    plt.rcParams.update({"xtick.bottom": True, "ytick.left": True})
    palette = sns.color_palette('Paired')
    for qsar, pcm in [('QSAR_results_regression.txt', 'PCM_results_regression.txt'),
                      ('QSAR_results_classification.txt', 'PCM_results_classification.txt')
                      ]:
        # Load data
        data_qsar = pd.read_csv(os.path.join(folder, qsar), sep='\t')
        data_pcm = pd.read_csv(os.path.join(folder, pcm), sep='\t')

        data_qsar.columns = ['Accession', 'Result'] + data_qsar.columns[2:].tolist()
        ids = ['Accession', 'Result']
        data_pcm.columns = ['Result'] + data_pcm.columns[1:].tolist()
        data_pcm.insert(0, 'Accession', ['PCM'] * data_pcm.shape[0])
        # Extract accession values
        data_qsar.loc[data_qsar.index, 'Accession'] = data_qsar['Accession'].str.replace('_WT', '')
        # Concatenate data
        data = pd.concat([data_qsar, data_pcm], axis=0)
        # Drop numbers
        if 'number' in data.columns:
            data = data.drop(columns=['number'])
        else:
            data = data.drop(columns=[col for col in data.columns if ':' in col])
        if 'regression' in qsar:
            # Separate cross-validation from test set results
            test_results = data[data['Result'] == 'Test set']
            cv_results = data[~data['Result'].isin(['Mean', 'SD', 'Test set'])]
            # Unpivot data
            test_results_melt = test_results.melt(ids,
                                                  ['R2', 'RMSE', 'MAE', 'Max Error'], 'Property')
            cv_results_melt = cv_results.melt(ids,
                                              ['R2', 'RMSE', 'MAE', 'Max Error'], 'Property')
            # Plot results
            f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=dpi)
            # CV
            sns.boxplot(x='Property', y='value', hue='Accession', data=cv_results_melt,
                        flierprops=dict(markerfacecolor='0.50', markersize=2), ax=ax1,
                        palette=palette, hue_order=['P30542', 'P29274', 'P29275', 'P0DMS8', 'P25099', 'P30543', 'P29276',
                                                    'P28647', 'Q60612', 'Q61618', 'P28190', 'PCM'])
            ax1.get_legend().remove()
            ax1.yaxis.grid(True)

            ax1.set_yticks(np.arange(-2.5, 4.6, 0.5))
            ax1.set_yticks(np.arange(-2.5, 4.6, 0.1), minor=True)
            ax1.set_ylim((-2.5, 4.5))
            ax1.set(ylabel='', xlabel='')
            # Test set
            sns.barplot(x='Property', y='value', hue='Accession', data=test_results_melt, ax=ax2,
                        linewidth=1, edgecolor=".2",
                        palette=palette, hue_order=['P30542', 'P29274', 'P29275', 'P0DMS8', 'P25099', 'P30543', 'P29276',
                                                    'P28647', 'Q60612', 'Q61618', 'P28190', 'PCM'])
            show_values_on_bars(ax2, space=0.2)
            # ax3 = ax2.twinx()
            ax2.yaxis.grid(True)
            ax2.set_yticks(np.arange(-15.0, 8.5, 0.5))
            ax2.set_yticks(np.arange(-15.0, 8.5, 0.1), minor=True)
            ax2.set_ylim((-3.5, 8.0))
            ax2.set(ylabel='', xlabel='')
        else:
            # Separate cross-validation from test set results
            test_results = data[data['Result'] == 'Test set']
            cv_results = data[~data['Result'].isin(['Mean', 'SD', 'Test set'])]
            # Unpivot data
            test_results_melt = test_results.melt(ids,
                                                  ['MCC', 'BACC', 'Sensitivity', 'Specificity', 'AUC N'], 'Property')
            cv_results_melt = cv_results.melt(ids,
                                              ['MCC', 'BACC', 'Sensitivity', 'Specificity', 'AUC N'], 'Property')
            # Plot results
            f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=dpi)
            # CV
            sns.boxplot(x='Property', y='value', hue='Accession', data=cv_results_melt,
                        flierprops=dict(markerfacecolor='0.50', markersize=2), ax=ax1,
                        palette=palette, hue_order=['P30542', 'P29274', 'P29275', 'P0DMS8', 'P25099', 'P30543', 'P29276',
                                                    'P28647', 'Q60612', 'Q61618', 'P28190', 'PCM'])
            ax1.get_legend().remove()
            ax1.yaxis.grid(True)

            ax1.set_yticks(np.arange(-0.7, 1.05, 0.1))
            ax1.set_yticks(np.arange(-0.7, 1.05, 0.025), minor=True)
            ax1.set_ylim((-0.7, 1.05))
            ax1.set(ylabel='', xlabel='')
            # Test set
            sns.barplot(x='Property', y='value', hue='Accession', data=test_results_melt, ax=ax2,
                        linewidth=1, edgecolor=".2",
                        palette=palette, hue_order=['P30542', 'P29274', 'P29275', 'P0DMS8', 'P25099', 'P30543', 'P29276',
                                                    'P28647', 'Q60612', 'Q61618', 'P28190', 'PCM'])
            show_values_on_bars(ax2)
            # ax3 = ax2.twinx()
            ax2.yaxis.grid(True)
            ax2.set_yticks(np.arange(-0.25, 1.05, 0.05))
            ax2.set_yticks(np.arange(-0.25, 1.05, 0.025), minor=True)
            ax2.set_ylim((-0.30, 1.05))
            ax2.set(ylabel='', xlabel='')
        # Move legend outside
        f.tight_layout()
        handles, labels = ax2.get_legend_handles_labels()
        legend = plt.legend(handles=handles, labels=["Human ADORA1", "Human ADORA2A", "Human ADORA2B", "Human ADORA3",
                                                     "Rat ADORA1", "Rat ADORA2A", "Rat ADORA2B", "Rat ADORA3",
                                                     "Mouse ADORA1", "Mouse ADORA3", "Bovine ADORA1", "PCM"],
                            bbox_to_anchor=(1.005, 1.2), loc=2, borderaxespad=0.)
        # for ax in [ax1, ax2]:
        #     for tick in ax.get_xticklabels():
        #         tick.set_rotation(45)
        # Show
        # plt.show()
        plt.savefig(f'{"regression" if "regression" in qsar else "classification"}_plot3.png',
                    bbox_extra_artists=(legend,), bbox_inches='tight')
