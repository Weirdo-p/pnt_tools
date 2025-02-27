from cProfile import label
from copyreg import add_extension
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from matplotlib.gridspec import GridSpec
import glob
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker

def RMS(gt, obs):
    rms = np.sqrt(((gt - obs) ** 2).sum() / gt.shape[0])
    return rms

font={
    #   'family':'Cambria',
      'size': 14, # corresponding to 10 pt
      'weight': 'bold'
}
font1={
    #   'family':'Cambria',
      'size': 12, # corresponding to 10 pt
      'weight': 'bold'
}

color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
         2: [(255 / 255), (102 / 255), (102 / 255)],  # red
         1: [(255 / 255), (204 / 255), (102 / 255)],  # blue
         0: [(20 / 255), (169 / 255), (89 / 255)],
         4: [(20 / 255), (169 / 255), (89 / 255)],
         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
marker = ['o', 's', '^']

direc=["E", "N", "U"]
## for tj format


def ploterror(time, neu, save, attribute, isSubplot):
    if neu.shape[0] < 1:
        return
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
    direc = attribute['legend']
    color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
         2: [(255 / 255), (102 / 255), (102 / 255)],  # red
         1: [(255 / 255), (146 / 255), (0 / 255)],  # blue
         0: [(0 / 255), (141 / 255), (0 / 255)],
         4: [(20 / 255), (169 / 255), (89 / 255)],
         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
    for i in range(neu.shape[1]):

        plt.grid(linestyle='-', color='k', alpha=0.2)
        direc_ = neu[:, i]
        plt.plot(time, (neu[:, i]) , ls="-", color=color[i], label=direc[i], linewidth=2.5)#, marker=marker[i], markersize=4)

        print(RMS(np.zeros(direc_.shape), direc_), " m")
    plt.ylabel(attribute['ylabel'], labelpad=3, fontsize = 13, fontdict=font)
    plt.xticks(size = 12)
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    plt.margins(x=0, y=0)


    if isSubplot:
        subPlotAtt = attribute["subplot"]
        xpos, ypos, width, height = subPlotAtt["xpos"], subPlotAtt["ypos"], subPlotAtt["width"], subPlotAtt["height"]
        axins = ax.inset_axes((xpos, ypos, width, height))
        rangeS, rangeE = subPlotAtt["range"][0], subPlotAtt["range"][1]
        for i in range (3):
            axins.plot(time[rangeS: rangeE], (neu[rangeS: rangeE, i]) , ls="-", color=color[i], label=direc[i], linewidth=2)#, marker=marker[i], markersize=4)
        
        ylimS, ylimE = subPlotAtt["ylim"][0], subPlotAtt["ylim"][1]
        axins.set_xlim(subPlotAtt["xlim"][0], subPlotAtt["xlim"][1])
        axins.set_ylim(ylimS, ylimE)
        loc1, loc2 = subPlotAtt["loc"][0], subPlotAtt["loc"][1]
        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec='k', lw=1)

    ax = plt.gca()
    if attribute["scientific"]:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((0,0)) 
        ax.yaxis.set_major_formatter(formatter)

    if attribute["xlim"][0] != attribute["xlim"][1]:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["ylim"][0] != attribute["ylim"][1]:
        plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel(attribute["xlabel"], fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()


def ploterror_paper(time, neu, save, attribute, isSubplot):
    if neu.shape[0] < 1:
        return
    xmajorFormatter = FormatStrFormatter('%f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(4.7, 1.4645669))
    direc = attribute['legend']
    # print(save)
    for i in range(neu.shape[1]):

        # ax = plt.subplot(3, 1, i + 1)
        # plt.tight_layout()
        # if i != 2:
        #     plt.setp(ax.get_xticklabels(), visible=False)
        plt.grid(linestyle='-', color='grey', alpha=0.4)
        direc_ = neu[:, i]
        # plt.plot(time, (neu[:, i]) , ls="-", color=color[i], label=direc[i], linewidth=2)#, marker=marker[i], markersize=4)
        plt.scatter(time, (neu[:, i]), color=color[i], label=direc[i], linewidth=2, s=0.6, marker='s')#, marker=marker[i], markersize=4)
        print(RMS(np.zeros(direc_.shape), direc_), " m")
    plt.ylabel(attribute['ylabel'], labelpad=3, fontsize = 13, fontdict=font)
    plt.yticks([-2, -1, 0, 1, 2],size = 12, fontproperties='Cambria')
    # plt.ylim(-3, 3)
    # legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)

    if isSubplot:
        subPlotAtt = attribute["subplot"]
        xpos, ypos, width, height = subPlotAtt["xpos"], subPlotAtt["ypos"], subPlotAtt["width"], subPlotAtt["height"]
        axins = ax.inset_axes((xpos, ypos, width, height))
        rangeS, rangeE = subPlotAtt["range"][0], subPlotAtt["range"][1]
        for i in range (3):
            axins.plot(time[rangeS: rangeE], (neu[rangeS: rangeE, i]) , ls="-", color=color[i], label=direc[i], linewidth=2)#, marker=marker[i], markersize=4)
        # axins.grid(b=False, linestyle='--', color='k', alpha=0.4)
        
        ylimS, ylimE = subPlotAtt["ylim"][0], subPlotAtt["ylim"][1]
        axins.set_xlim(subPlotAtt["xlim"][0], subPlotAtt["xlim"][1])
        axins.set_ylim(ylimS, ylimE)
        # mark_inset()
        loc1, loc2 = subPlotAtt["loc"][0], subPlotAtt["loc"][1]
        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec='k', lw=1)
    # ax = legend.get_frame()
    # ax.set_alpha(1)
    # ax.set_facecolor('none')

    ax = plt.gca()
    
    if attribute["scientific"]:
        # scientific expression
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((0,0)) 
        ax.yaxis.set_major_formatter(formatter)

    # plt.xlim(0, 12000)
    xmajorLocator  = MultipleLocator(120)
    ymajorLocator  = MultipleLocator(5)

    # ax.xaxis.set_major_locator(xmajorLocator) 
    ax.yaxis.set_minor_locator(MultipleLocator(0.5)) 
    # plt.yticks()

    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # # plt.xlim(0, 60)
    # label = []
    # test = plt.xticks(size = 12, fontproperties='Cambria')
    # for i in range(0, len(test[0])):
    #     label.append("{:.0f}".format(test[0][i] / 60))
    
    # plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    # plt.margins(x=0, y=0)
    if attribute["xlim"][0] != attribute["xlim"][1]:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["ylim"][0] != attribute["ylim"][1]:
        plt.ylim(attribute["ylim"][0], attribute["ylim"][1])
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel(attribute["xlabel"], fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()

def plotTraj(trajs={ }, save='./'):
    """plot trajectories
       
    Args:
        trajs (list, optional): _description_. Defaults to [].
        each one is a traj(in enu-frame)
    """
    plt.figure(figsize=(5.0393701, 5.0393701))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(linestyle='-', color=grey, alpha=0.5, linewidth=1)
    i = 0
    for key in trajs.keys():
        plt.plot(trajs[key][:, 0], trajs[key][:, 1], label=key, color=color[i])
        i += 1
    
    test = plt.xticks(size = 12, fontproperties='Cambria')

    plt.yticks(size = 12, fontproperties='Cambria')
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.11), fancybox=False)
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.09, top=0.91, wspace=0.01, hspace=0.1)
    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')
    ax = plt.gca()
    xmajorLocator  = MultipleLocator(20)
    ymajorLocator  = MultipleLocator(20)
    ax.set_aspect(1)
    plt.ylim(-6, 6)
    plt.xlim(-6, 6)

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.ylabel("m", labelpad=3, fontsize = 13, fontdict=font)
    plt.xlabel("m", labelpad=3, fontsize = 13, fontdict=font)
    plt.margins(x=0, y=0.00)
    plt.savefig(save, transparent=True)
    plt.show()


def plotBox(neu, save):
    plt.figure(figsize=(5.0393701, 3.4645669))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(linestyle='-', color=grey, alpha=0.5, linewidth=1)
    
    test = []
    # test.
    for key in neu.keys():
        test.append(neu[key])
    plt.boxplot(test, showfliers=True, 
                labels=neu.keys(), whis=1.5, flierprops={'marker':'+', 'markeredgecolor': 'red'},
                medianprops={'color': 'green'}
                )
        
    plt.ylim(0, 1)
    plt.margins(x=0, y=0)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.ylabel("Error(m)", labelpad=3, fontsize = 13, fontdict=font)
    plt.xticks(size = 12, fontproperties='Cambria')
    plt.yticks(size = 12, fontproperties='Cambria')
    # plt.legend()
    plt.savefig(save, transparent=True)
    plt.show()
    # pass

def plotSatNum(satNum, save):
    plt.figure(figsize=(5.0393701, 3.4645669))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(b=True, linestyle='-', color=grey, alpha=0.5, linewidth=1)
    
    plt.scatter(range(0, satNum.shape[0]), satNum, linewidths=1.5, c=color[1],s=5, label="G+C+E")
    # legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # ax = legend.get_frame()
    # ax.set_alpha(1)
    # ax.set_facecolor('none')

    plt.margins(x=0, y=0)
    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    ax = plt.gca()
    xmajorLocator  = MultipleLocator(1500)
    ymajorLocator  = MultipleLocator(5)

    ax.xaxis.set_major_locator(xmajorLocator) 

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    
    label = []
    test = plt.xticks(size = 12, fontproperties='Cambria')
    for i in range(0, len(test[0])):
        label.append("{:.1f}".format(test[0][i] / 60 / 10))
    plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    plt.xlim(0, 7500)
    plt.ylim(0, 30)

    plt.ylabel("Number of Satellite", labelpad=3, fontsize = 13, fontdict=font)
    plt.xlabel("Epoch (min)", fontdict=font)

    plt.savefig(save, transparent=True)
    plt.show()
    

def plotPdop(time, pdop, nsat, save):
    # color = {3: [(63 / 255), (169 / 255), (245 / 255)],  # black
    #         2: [(212 / 255), (86 / 255), (46 / 255)],
    #         1: [(219 / 255), (180 / 255), (40 / 255)],  # blue
    #         0: [(132 / 255), (186 / 255), (66 / 255)],
    #         4: [(20 / 255), (169 / 255), (89 / 255)],
    #         5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
    plt.figure(figsize=(5.0393701, 3.3))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.grid(linestyle='-', color='k', alpha=0.1, linewidth=0.5)
    
    plt.scatter(time, pdop, linewidths=1.5, color=color[1],s=10, marker="o")
    plt.ylabel("PDOP", fontdict=font)
    plt.margins(x=0, y=0)
    plt.subplots_adjust(left=0.13, right=0.94, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    # plt.ylim(0, 20)
    ax = plt.gca()
    ax.spines['left'].set_color([(238 / 255), (159 / 255), (0 / 255)])
    # plt.yticks([0, 5, 10, 15, 20], size=12)
    plt.xticks(size=12)
    # ax.yaxis.set_minor_locator(MultipleLocator(2.5)) 
    for i in range(len(ax.get_yticklabels())):
        ax.get_yticklabels()[i].set_color([(238 / 255), (159 / 255), (0 / 255)])
    plt.ylim(0, 8)
    ax.tick_params(width=1.5, axis='y', which='minor',colors=[(238 / 255), (159 / 255), (0 / 255)])
    ax.tick_params(width=1.5, axis='y', color=[(238 / 255), (159 / 255), (0 / 255)])
    ax.tick_params(width=1.5, axis='x')
    plt.yticks(size=12)
    plt.xlabel("time (s)", fontdict=font)
    plt.twinx()
    plt.scatter(time, nsat, linewidths=1.5, color=color[0],s=10, marker="^")
    ax = plt.gca()
    for i in range(len(ax.get_yticklabels())):
        ax.get_yticklabels()[i].set_color(color[0])
    # ax.tick_params(width=1.5, which='minor',colors=color[0])
    ax.tick_params(width=1.5, axis='y', color=color[0])
    ax.spines['right'].set_color(color[0])
    ax.spines['left'].set_color([(238 / 255), (159 / 255), (0 / 255)])
    # ax.xaxis.set_major_locator(xmajorLocator) 
    # ax.yaxis.set_minor_locator(MultipleLocator(2.5)) 
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    # plt.yticks([0, 10, 20, 30], size=12)
    # ax.yaxis.set_minor_locator(MultipleLocator(5)) 
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.ylim(10, 40)
    plt.xlim(0, 2000)

    # plt.ylim(0, 30)
    # plt.xlim(0, 200)
    # label = []
    # test = plt.xticks(size = 12, fontproperties='Cambria')
    # for i in range(0, len(test[0])):
    #     label.append("{:.1f}".format(test[0][i] / 60 / 10))
    # plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    # plt.xlim(0, 7500)

    # plt.ylabel("PDOP", labelpad=3, fontsize = 13, fontdict=font)
    # plt.xlabel("Epoch (min)", fontdict=font)
    # plt.ylim(0, 8)
    plt.margins(x=0, y=0)
    plt.xlabel("time (s)", fontdict=font)

    plt.savefig(save, transparent=True)
    plt.show()

color = {3: [(0 / 255), (167 / 255), (251 / 255)],  # black
        2: [(252 / 255), (169 / 255), (0 / 255)],  # red
        1: [(100 / 255), (13 / 255), (171 / 255)],  # blue
        0: [(245 / 255), (82 / 255), (0 / 255)],
        4: [(20 / 255), (169 / 255), (89 / 255)],
        5: [(70 / 255), (114 / 255), (196 / 255)]}  # green
def plotCDF(cdf_dict = {}, save="", color=color):
    plt.figure(figsize=(4, 4))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(linestyle='-', color='k', alpha=0.17)
    plt.margins(x=0, y=0)
    i = 0
    for key in cdf_dict.keys():
        cdf_dict[key] = np.sort(np.fabs(cdf_dict[key]))
        prob = np.array(range(0, cdf_dict[key].shape[0])) / cdf_dict[key].shape[0]
        print(cdf_dict[key][prob > 0.95][0])
        plt.plot(cdf_dict[key], np.array(range(0, cdf_dict[key].shape[0])) / cdf_dict[key].shape[0], label=key, color=color[i], linewidth=3)
        i+=1

    # legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.155), fancybox=False)
    
    plt.subplots_adjust(left=0.13, right=0.96, bottom=0.13, top=0.89, wspace=0.01, hspace=0.1)
    # ax = legend.get_frame()
    # ax.set_alpha(1)
    # ax.set_facecolor('none')
    ax = plt.gca()
    # ax.set_aspect(1)

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    # plt.xlabel("Error (m)", fontdict=font)
    # plt.ylabel("Probability", labelpad=3, fontsize = 13, fontdict=font)
    plt.ylim(0, 1)
    plt.xlim(0, 40)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    
    plt.savefig(save, transparent=True)
    plt.show()
    
def plotBar(bar_dict={}, save="./"):
    # 目前仅实现了三个为一组的柱状图
    plt.figure(figsize=(5.0393701, 3.4645669))
    grey = [(191 / 255), (191 / 255), (191 / 255)]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(b=True, linestyle='--', color='k', alpha=0.5)
    
    x = np.arange(len(bar_dict.keys()))
    width = 0.25
    data = []
    for key in bar_dict.keys():
        data.append(bar_dict[key])
    data = np.array(data)
    
    labels = ["E", "N", "U"]
    print(data)
    print(x)
    i, j = -1, 0
    for key in bar_dict.keys():
        plt.bar(x + i * width, data[:, j], width, label=labels[j], color=color[j])
        i += 1
        j += 1
    plt.xticks(x, labels=bar_dict.keys(), size = 12, fontproperties='Cambria')
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.155), fancybox=False)
    
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.1, top=0.89, wspace=0.01, hspace=0.1)
    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    # plt.xticks(size = 12, fontproperties='Cambria')
    plt.ylabel("Error (m)", labelpad=3, fontsize = 13, fontdict=font)
    plt.ylim(0, 14)
    
    plt.savefig(save, transparent=True)
    
    plt.show()
    

def plotErrorWithCov(time, data, std, direc, save):
    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(5.0393701, 3.4645669))
    # direc = ["Y", "P", "R"]
    marker=["o", "^", "D"]
    # print(save)

    # ax = plt.subplot(3, 1, i + 1)
    # plt.tight_layout()
    # if i != 2:
    #     plt.setp(ax.get_xticklabels(), visible=False)
    plt.grid(b=True, linestyle='--', color='k', alpha=0.5)
    # direc_ = direc
    plt.plot(time, data , ls="-", color="b", label=direc, linewidth=2)#, marker=marker[i], markersize=4)

    plt.plot(time, std * 3, linewidth=1.5, ls="--", color="r", label="3-sigma")
    plt.plot(time, -std * 3, linewidth=1.5, ls="--", color="r")

    # print(RMS(np.zeros(direc_.shape), direc_), " m")
    plt.ylabel("Error(m)", labelpad=3, fontsize = 13, fontdict=font)
    plt.yticks(size = 12, fontproperties='Cambria')
    # plt.ylim(-3, 3)
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)

    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')

    ax = plt.gca()
    # plt.xlim(0, 12000)
    # xmajorLocator  = MultipleLocator(1500)
    # ymajorLocator  = MultipleLocator(5)

    # ax.xaxis.set_major_locator(xmajorLocator) 
    # ax.yaxis.set_major_locator(ymajorLocator) 

    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # plt.xlim(0, 60)
    # label = []
    # test = plt.xticks(size = 12, fontproperties='Cambria')
    # for i in range(0, len(test[0])):
    #     label.append("{:.1f}".format(test[0][i] / 60 / 10))
    
    # plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    # plt.margins(x=0, y=0)
    plt.xlim(0, 60)
    plt.ylim(-8, 8)
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()


def CompareCov(time_data1, time_data2, data1, data2, label1, label2, save):
    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(5.0393701, 3.4645669))
    # direc = ["Y", "P", "R"]
    marker=["o", "^", "D"]
    # print(save)

    # ax = plt.subplot(3, 1, i + 1)
    # plt.tight_layout()
    # if i != 2:
    #     plt.setp(ax.get_xticklabels(), visible=False)
    plt.grid(b=True, linestyle='--', color='k', alpha=0.5)
    # direc_ = direc
    plt.plot(time_data1, data1, ls="-", color="b", label=label1, linewidth=2)#, marker=marker[i], markersize=4)

    plt.plot(time_data2, data2, ls="-", color="r", label=label2, linewidth=2)

    # print(RMS(np.zeros(direc_.shape), direc_), " m")
    plt.ylabel("Error(m)", labelpad=3, fontsize = 13, fontdict=font)
    plt.yticks(size = 12, fontproperties='Cambria')
    # plt.ylim(-3, 3)
    legend = plt.legend(loc='upper right', fontsize = 12, edgecolor='black', numpoints=1, ncol=3, prop=font1, bbox_to_anchor=(1.02, 1.16), fancybox=False)
    # plt.subplots_adjust(top=1)
    plt.margins(x=0, y=0)

    ax = legend.get_frame()
    ax.set_alpha(1)
    ax.set_facecolor('none')

    ax = plt.gca()
    # plt.xlim(0, 12000)
    # xmajorLocator  = MultipleLocator(1500)
    # ymajorLocator  = MultipleLocator(5)

    # ax.xaxis.set_major_locator(xmajorLocator) 
    # ax.yaxis.set_major_locator(ymajorLocator) 

    # ax.xaxis.set_major_formatter(xmajorFormatter)
    # plt.xlim(0, 60)
    # label = []
    # test = plt.xticks(size = 12, fontproperties='Cambria')
    # for i in range(0, len(test[0])):
    #     label.append("{:.1f}".format(test[0][i] / 60 / 10))
    
    # plt.xticks(test[0], label, size = 12, fontproperties='Cambria')
    # plt.margins(x=0, y=0)
    plt.xlim(0, 60)
    # plt.ylim(-8, 8)
    # print(test)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()

def plotMapPoint(points,save=""):
    xmajorFormatter = FormatStrFormatter('%1.2f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(5.0393701, 5.0393701))
    # direc = ["Y", "P", "R"]
    marker=["o", "^", "D"]
    # print(save)

    # ax = plt.subplot(3, 1, i + 1)
    # plt.tight_layout()
    # if i != 2:
    #     plt.setp(ax.get_xticklabels(), visible=False)
    # plt.grid(b=False, linestyle='--', color='k', alpha=0.5)
    plt.scatter(points[:, 1], -points[:,0], s=0.5, c=[(226 / 255), (221 / 255), (69 / 255)],alpha=0.8)
    
    plt.margins(x=0, y=0)
    # a={}

    ax = plt.gca()
    ax.set_aspect(1)

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.yticks([])
    plt.xticks([])
    plt.ylim(-20, 120)
    plt.xlim(-80, 80)

    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    # plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()

def plotPointsWithTraj(trajs={}, points={}, save="./"):

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.figure(figsize=(3.4645669, 3.4645669), projection='3d')
    # direc = ["Y", "P", "R"]
    marker=["o", "^", "D"]
    # print(save)

    # ax = plt.subplot(3, 1, i + 1)
    # plt.tight_layout()
    # if i != 2:
    #     plt.setp(ax.get_xticklabels(), visible=False)
    # plt.grid(b=False, linestyle='--', color='k', alpha=0.5)
    trajs = {}
    i = 0
    for key, value in trajs.items():
        plt.plot(value[:, 0], value[:, 1], value[:, 2], s=0.5, c=color[i], label=key)
        i += 1
    
    for key, value in trajs.items():
        plt.scatter(value[:, 0], value[:, 1], value[:, 2], linewidth=2, c=color[i], label=key)
    
    plt.margins(x=0, y=0)
    # a={}

    ax = plt.gca()
    ax.set_aspect(1)

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.yticks([])
    plt.xticks([])
    plt.ylim(-20, 120)
    plt.xlim(-80, 80)

    plt.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    # plt.xlabel("Epoch (sec)", fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()

def plotamb(time, neu, save, attribute, isSubplot):
    xmajorFormatter = FormatStrFormatter('%f') #设置x轴标签文本的格式 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(1, 1, figsize=(5.0393701, 3.4645669))
    direc = attribute['legend']

    plt.grid(linestyle='-', color='grey', alpha=0.4)
    plt.scatter(time, (neu), color=color[0], label=direc, linewidth=2, s=0.6, marker='s')#, marker=marker[i], markersize=4)
    plt.ylabel(attribute['ylabel'], labelpad=3, fontsize = 13, fontdict=font)
    plt.margins(x=0, y=0)

    ax = plt.gca()
    
    if attribute["scientific"]:
        # scientific expression
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((0,0)) 
        ax.yaxis.set_major_formatter(formatter)

    if attribute["xlim"][0] != attribute["xlim"][1]:
        plt.xlim(attribute["xlim"][0], attribute["xlim"][1])
    if attribute["ylim"][0] != attribute["ylim"][1]:
        plt.ylim(attribute["ylim"][0], attribute["ylim"][1])

    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    plt.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.89, wspace=0.01, hspace=0.1)
    plt.xlabel(attribute["xlabel"], fontdict=font)
    plt.savefig(save, transparent=True)
    plt.show()
