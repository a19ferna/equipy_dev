import matplotlib.pyplot as plt

def _list_to_plot(values):
    substraction_list = [values[0]]
    for i in range(len(values)-1):
        substraction_list.append(values[i+1]-values[i])
    substraction_list.append(values[-1])
    return substraction_list

def _get_bottom(values):
    bottom = [0]
    for i in range(len(values)-1):
        bottom.append(values[i])
    bottom.append(0)
    return bottom

def _set_colors(substraction_list):
    bar_colors = ['tab:grey']
    for i in range(1,len(substraction_list)-1):
        if substraction_list[i] > 0:
            bar_colors.append('tab:orange')
        else:
            bar_colors.append('tab:green')
    bar_colors.append('tab:grey')

    return bar_colors

def _add_bar_labels(values, pps, ax):    
    true_values = values + (values[-1],)

    for i, p in enumerate(pps):
        height = true_values[i]
        ax.annotate('{}'.format(height),
            xy=(p.get_x() + p.get_width() / 2, height),
            xytext=(0, 3), 
            textcoords="offset points",
            ha='center', va='bottom')
        
def _add_doted_points(ax, values):
    for i, v in enumerate(values):
        ax.plot([i+0.25, i+1.25], [v, v],
                linestyle='--', linewidth=1.5, c='grey')
        
def waterfall_plot(unfs_levels, ax=None, hatch=False, approximate=False):

    (keys,values) = zip(*unfs_levels.items())

    if ax == None:
        fig, ax = plt.subplots()

    substraction_list = _list_to_plot(values)
    bar_colors = _set_colors(substraction_list)

    leg = keys + ('Final Model',)

    if not hatch:
        pps = ax.bar(leg, substraction_list, color=bar_colors, edgecolor='k', bottom=_get_bottom(values))
        
    
        _add_bar_labels(values, pps, ax)
        _add_doted_points(ax, values)

        ax.set_ylabel(f'Unfairness in A_{keys[-1]}')
        ax.set_ylim(0,1.1)
        if approximate: 
            ax.set_title(f'Sequential (approximate) fairness: $\\nu_{keys[-1]}$ result')
        else:
            ax.set_title(f'Sequential (exact) fairness: $\\nu_{keys[-1]}$ result')
        
        plt.show()
    
    else:
        ax.bar(leg, substraction_list, color='w', edgecolor=bar_colors, bottom=_get_bottom(values), hatch='//')

def waterfall_plot_approximate(unfs_levels):

    waterfall_plot(unfs_levels[0])

    for i in range(1, len(unfs_levels)):
        unfs = (unfs_levels[0], unfs_levels[i])
        
        fig, ax = plt.subplots()   
        
        for i, dict in enumerate(unfs): 

            if i==0:
                waterfall_plot(dict, ax=ax, hatch=True)

            else:
                waterfall_plot(dict, ax=ax, approximate=True)


        plt.show()
