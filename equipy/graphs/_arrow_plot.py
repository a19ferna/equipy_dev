import numpy as np
import matplotlib.pyplot as plt

def arrow_plot(unfs_dict, performance_dict, permutations=False, base_model=True, final_model=True):
    """
    Generates an arrow plot representing the fairness-performance combinations for each level of fairness.

    Parameters:
    - unfs_dict (dict): A dictionary containing unfairness values associated to the sequentally fair output datasets.
    - performance_dict (dict): A dictionary containing performance values associated to the sequentally fair output datasets.
    - permutations (bool, optional): If True, displays permutations of arrows based on input dictionaries.
                                     Defaults to False.
    - base_model (bool, optional): If True, includes the base model arrow. Defaults to True.
    - final_model (bool, optional): If True, includes the final model arrow. Defaults to True.

    Returns:
    None

    Plotting Conventions:
    - Arrows represent different fairness-performance combinations.
    - Axes are labeled for unfairness (x-axis) and performance (y-axis).

    Note:
    - This function uses global variable `ax` for plotting, ensuring compatibility with external code.
    """
    x = []
    y = []
    sens = [0]

    for i, key in enumerate(unfs_dict.keys()):
        x.append(unfs_dict[key])
        if i != 0:
            sens.append(int(key[9:]))

    for key in performance_dict.keys():
        y.append(performance_dict[key])

    global ax

    if not permutations:
        fig, ax = plt.subplots()

    line = ax.plot(x, y, linestyle="--", alpha=0.25, color="grey")[0]

    for i in range(len(sens)):
        if (i == 0) & (base_model):
            line.axes.annotate(f"Base\nmodel", xytext=(
                x[0]+np.min(x)/20, y[0]), xy=(x[0], y[0]), size=10)
            ax.scatter(x[0], y[0], label="Base model", marker="^", s=100)
        elif i == 1:
            label = f"$A_{sens[i]}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        elif (i == len(x)-1) & (final_model):
            
            label = f"$A_{1}$" + r"$_:$" + f"$_{i}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="*", s=150)
        elif (i == 2) & (i < len(x)-1):
            
            label = f"$A_{sens[1]}$" + r"$_,$" + f"$_{sens[i]}$-fair"
            line.axes.annotate(label, xytext=(
                x[i]+np.min(x)/20, y[i]), xy=(x[i], y[i]), size=10)
            ax.scatter(x[i], y[i], label=label, marker="+", s=150)
        else:
            ax.scatter(x[i], y[i], marker="+", s=150, color="grey", alpha=0.4)
    ax.set_xlabel("Unfairness")
    ax.set_ylabel("Performance")
    ax.set_xlim((np.min(x)-np.min(x)/10-np.max(x)/10,
                np.max(x)+np.min(x)/10+np.max(x)/10))
    ax.set_ylim((np.min(y)-np.min(y)/10-np.max(y)/10,
                np.max(y)+np.min(y)/10+np.max(y)/10))
    ax.set_title("Exact fairness")
    ax.legend(loc="best")


def arrow_plot_permutations(unfs_list, performance_list):
    """
    Plot arrows representing the fairness-performance combinations for each level of fairness for all permutations (order of sensitive variables which with fairness is calculate).

    Parameters:
    - unfs_list (list): A list of dictionaries containing unfairness values for each permutation of fair output datasets.
    - performance_list (list): A list of dictionaries containing performance values for each permutation of fair output datasets.

    Returns:
    None

    Plotting Conventions:
    - Arrows represent different fairness-performance combinations for each scenario in the input lists.
    - Axes are labeled for unfairness (x-axis) and performance (y-axis).

    Example Usage:
    >>> arrow_plot_permutations(unfs_list, performance_list)

    Note:
    - This function uses global variable `ax` for plotting, ensuring compatibility with external code.
    """
    global ax
    fig, ax = plt.subplots()
    for i in range(len(unfs_list)):
        if i == 0:
            arrow_plot(unfs_list[i], performance_list[i],
                       permutations=True, final_model=False)
        elif i == len(unfs_list)-1:
            arrow_plot(unfs_list[i], performance_list[i],
                       permutations=True, base_model=False)
        else:
            arrow_plot(unfs_list[i], performance_list[i], permutations=True,
                       base_model=False, final_model=False)