import matplotlib.pyplot as plt
import argparse


def drawnHistogram(label_list, name_list, color_list, hatch_list, figurename, xlabel, ylabel, *arg):
    plt.figure(figsize=(18,7))

    x = list(range(len(name_list)))
    name = []
    totalwidth, n = 0.5, len(arg)
    for i in range(len(x)):
        name.append(x[i] + totalwidth * (n - 1) / (2 * n) / 0.7)

    plt.xticks(name, name_list, rotation=0)
    width = totalwidth / (0.6 * n)
    temp_max = 0
    for i in range(len(label_list)):
        a = plt.bar(x, arg[i], width=width, label=label_list[i], color=color_list[i], hatch=hatch_list[i])
        if temp_max < max(arg[i]):
            temp_max = max(arg[i])
        for j in range(len(x)):
            x[j] = x[j] + width








    plt.ylim(0, 105)



    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(xlabel, size=20, labelpad=10)
    plt.ylabel(ylabel, size=24)
    plt.tick_params(length=5, pad=5, labelsize=20)



    plt.tight_layout()
    plt.savefig(figurename)
    plt.show()


def drawn_fig_2c(output=None):
    figure_name = output or 'Fig_2c.pdf'
    x_label = ''
    y_label = 'Quantitative Analysis Error Gap (%)'


    name_list = ['Grain', 'EMPS', 'AH', 'UHCS', 'MD', 'Super', 'EBC']
    label_list = ['segmentation of grains', 'segmentation of particles', 'segmentation of metallic phases (AH)', 'segmentation of metallic phases (UHCS)', 'segmentation of metallic phases (MetalDAM)', 'segmentation of metallic phases (Super)', 'segmentation of coating and defects']

    hatch_list = ['', '', '', '', '', '', '', '']
    color_list = ['brown', 'grey', 'goldenrod', 'seagreen', 'c', 'cornflowerblue', 'deepskyblue', 'lightblue']


    name_list = ['Accuracy', 'Precision', 'Recall', 'Dice', 'mIoU', 'HD95', 'NSD', 'CE']


    Grain = [9.4, 60.2, 0, 9.4, 9.4, 10.2, 16.2, 0]

    EMPS = [135.7, 135.7, 738, 135.7, 135.7, 0, 135.7, 105]
    AH = [0, 0, 3451, 17.5, 17.5, 17.5, 17.5, 0]
    UHCS = [0.8, 2.8, 0.8, 1.1, 1.1, 2.8, 0.8, 0]
    MD = [0, 0, 0, 0, 0, 4.8, 0, 4.8]
    Super = [1.46, 0, 0, 0, 0, 0, 0, 1.46]
    EBC = [0, 0, 3.1, 6.2, 6.2, 0, 0, 6.2]











    drawnHistogram(label_list, name_list, color_list, hatch_list, figure_name, x_label, y_label, Grain,
                  EMPS, AH, UHCS, MD, Super, EBC)


def drawn_fig_4(output=None):
    figure_name = output or 'Fig_4.pdf'
    x_label = ''
    y_label = 'Spearman Correlation Coefficient'

    name_list = ['G', 'E', 'A', 'U', 'M', 'S','E']

    label_list = ['best', 'add', 'tcrf']

    hatch_list = ['', '', '', '', '', '', '', '']

    color_list = ["salmon", "wheat", "skyblue", "plum"]

    best = [0.881, 0.874, 0.965, 0.762, 0.857, 0.976, 0.946]

    add = [0.636, 0.762, 0.762, 0.595, 0.810, 0.905, 0.755]
    tcrf = [0.906, 0.874, 0.797, 0.738, 0.833, 0.905, 0.976]


    drawnHistogram(label_list, name_list, color_list, hatch_list, figure_name, x_label, y_label, best, add, tcrf)


def drawn_fig_2d(output=None):
    figure_name = output or 'fig_2d.pdf'
    x_label = ''
    y_label = 'Top-3 Overlap Rate (%)'


    name_list = ['Grain', 'EMPS', 'AH', 'UHCS', 'MD', 'Super', 'EBC']
    label_list = ['segmentation of grains', 'segmentation of particles', 'segmentation of metallic phases (AH)', 'segmentation of metallic phases (UHCS)', 'segmentation of metallic phases (MetalDAM)', 'segmentation of metallic phases (Super)', 'segmentation of coating and defects']

    hatch_list = ['', '', '', '', '', '', '', '']
    color_list = ['brown', 'grey', 'goldenrod', 'seagreen', 'c', 'cornflowerblue', 'deepskyblue', 'lightblue']


    name_list = ['Accuracy', 'Precision', 'Recall', 'Dice', 'mIoU', 'HD95', 'NSD', 'CE']









    Grain = [33.33, 33.33, 66.66, 33.33, 33.33, 66.66, 33.33, 66.66]
    EMPS = [33.33, 33.33, 0, 66.66, 66.66, 33.33, 66.66, 66.66]
    AH = [66.66, 66.66, 0, 66.66, 66.66, 66.66, 66.66, 66.66]
    UHCS = [66.66, 33.33, 66.66, 66.66, 66.66, 66.66, 66.66, 66.66]
    MD = [66.66, 66.66, 66.66, 66.66, 66.66, 33.33, 66.66, 66.66]
    Super = [100, 100, 66.66, 100, 100, 100, 100, 66.66]
    EBC = [66.66, 33.33, 66.66, 66.66, 66.66, 66.66, 66.66, 66.66]

    drawnHistogram(label_list, name_list, color_list, hatch_list, figure_name, x_label, y_label, Grain,
                  EMPS, AH, UHCS, MD, Super, EBC)

def top_correlation_result(output=None):
    figure_name = output or 'top_Ablation_vs.pdf'

    x_label = 'Top-k'
    y_label = 'Jaccard Similarity'

    name_list = ['Top-1', 'Top-3']

    hatch_list = ['', '', '', '', '', '']
    color_list = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", 'c']

    label_list = ['mIoU', 'mIoU + BR', 'mIoU + OCC', 'BR + OCC', 'MBSS']

    mIoU = [0.7, 0.7]
    Dice = [0.8, 0.8]
    HD = [0.9, 0.9]
    MAE = [0.6, 0.6]
    MBSS = [0.9, 0.9]


    drawnHistogram(label_list, name_list, color_list, hatch_list, figure_name, x_label, y_label, mIoU,
                  Dice, HD, MAE, MBSS)


def ablation_result(output=None):
    figure_name = output or 'top_Ablation_vs.pdf'
    x_label = 'Metrics'

    y_label = 'Jaccard Similarity'

    name_list = ['mIoU', 'mIoU + BR', 'mIoU + OCC', 'BR + OCC', 'MBSS']
    name_list = ['5', '10', '15', '20']
    name_list = ['mIoU', 'mIoU+BR', 'mIoU+OCC', 'BR+OCC', 'MBSS']

    hatch_list = ['', '', '', '', '', '']
    color_list = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", 'c']

    label_list = ['Top-1', 'Top-3']


    top_1 = [0.7, 0.7, 0.7, 0.7, 0.7]
    top_3 = [0.8, 0.8, 0.8, 0.8, 0.8]


    drawnHistogram(label_list, name_list, color_list, hatch_list, figure_name, x_label, y_label, top_1,
                  top_3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw paper summary bar charts.")
    parser.add_argument(
        "--figure",
        choices=("fig2c", "fig4", "fig2d", "top-correlation", "ablation"),
        default="fig2d",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    functions = {
        "fig2c": drawn_fig_2c,
        "fig4": drawn_fig_4,
        "fig2d": drawn_fig_2d,
        "top-correlation": top_correlation_result,
        "ablation": ablation_result,
    }
    functions[args.figure](args.output)
