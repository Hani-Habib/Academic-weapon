import numpy as np
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_grid(opt_grid,title=''):
    """ utility over the budget shares, as a 3D and as a contour plot

    Args:

        opt_grid (SimpleNamespace): optimal budget shares and utility
        title (str): name of the calibration

    """

    fig = plt.figure(figsize=(14,5.5))

    # 3D plot
    ax = fig.add_subplot(1,2,1,projection='3d')

    ax.plot_surface(opt_grid.s1_grid,opt_grid.w_grid,opt_grid.u_grid,cmap='viridis')
    ax.plot([opt_grid.s1],[opt_grid.w],[opt_grid.u],'o',ms=9,color=colors[3],label='grid search optimum')

    ax.set_title(f'Utility({title})')
    ax.set_xlabel('$s_1$')
    ax.set_ylabel('$w$')
    ax.set_zlabel('utility, $u$')
    ax.legend(loc='upper left',fontsize=11)

    # contour plot
    ax = fig.add_subplot(1,2,2)

    cs = ax.contourf(opt_grid.s1_grid,opt_grid.w_grid,opt_grid.u_grid,levels=30)
    fig.colorbar(cs,ax=ax,label='utility, $u$')
    ax.plot(opt_grid.s1,opt_grid.w,'o',ms=9,color=colors[3],label='grid search optimum')

    ax.set_title(f'Contours ({title})')
    ax.set_xlabel('$s_1$')
    ax.set_ylabel('$w$')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc='upper right',fontsize=11)
    ax.grid(visible=False)

    fig.tight_layout()
    plt.show()


def plot_convergencepath(opt_grid,opt,title=''):
    """ the convergence path on the contour plot, and the distance to the end point

    Args:

        opt_grid (SimpleNamespace): a solution from ConsumerClass.solve_grid()
        opt (SimpleNamespace): a solution from ConsumerClass.solve()
        title (str): name of the calibration, put in the panel titles

    """

    fig = plt.figure(figsize=(14,5.5))

    # the path on top of the contours
    ax = fig.add_subplot(1,2,1)

    cs = ax.contourf(opt_grid.s1_grid,opt_grid.w_grid,opt_grid.u_grid,levels=30)
    fig.colorbar(cs,ax=ax,label='utility, $u$')

    ax.plot(opt.path[:,0],opt.path[:,1],'-o',ms=5,color=colors[3],label='path')
    ax.plot(opt.path[0,0],opt.path[0,1],'s',ms=9,color=colors[1],label='start')
    ax.plot(opt.s1,opt.w,'*',ms=15,color=colors[2],label='end')

    ax.set_title(f'Convergence path ({title})')
    ax.set_xlabel('$s_1$')
    ax.set_ylabel('$w$')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc='upper right',fontsize=11)
    ax.grid(visible=False)

    # distance to the end point
    ax = fig.add_subplot(1,2,2)

    dist = np.sqrt(np.sum((opt.path-opt.path[-1])**2,axis=1))

    ax.plot(np.arange(dist.size-1),dist[:-1],'-o',ms=5,color=colors[3])

    ax.set_yscale('log')
    ax.set_title(f'Distance to the end point ({title})')
    ax.set_xlabel('iteration')
    ax.set_ylabel('$\\|s_k - s_{end}\\|$')

    fig.tight_layout()
    plt.show()


def plot_revenue(tau_vec,R,names,calibrations):
    """ revenue against the tax rate, one panel per calibration

    Args:

        tau_vec (ndarray): the tax rates
        R (dict): revenue arrays, keyed by (calibration,instrument)
        names (list): the instruments to draw, in order
        calibrations (list): the calibrations, one panel each

    """

    fig,axes = plt.subplots(1,len(calibrations),figsize=(12,4.5),sharey=True)

    for ax,cal in zip(axes,calibrations):

        for j,name in enumerate(names):
            ax.plot(tau_vec,R[cal,name],color=f'C{j}',label=name)

        ax.set_xlabel(r'tax rate, $\tau$')
        ax.set_title(cal)
        ax.grid(True)

    axes[0].set_ylabel('revenue, $R$')
    axes[0].legend(fontsize=8)

    fig.tight_layout()


def plot_utility_revenue(R,u,names,calibrations):
    """ utility against the revenue raised, one panel per calibration

    Args:

        R (dict): revenue arrays, keyed by (calibration,instrument)
        u (dict): utility arrays, same keys
        names (list): the instruments to draw, in order
        calibrations (list): the calibrations, one panel each

    """

    fig,axes = plt.subplots(1,len(calibrations),figsize=(12,4.5),sharey=True)

    for ax,cal in zip(axes,calibrations):

        for j,name in enumerate(names):
            ls = '--' if name == 'Lump-sum' else '-' # the benchmark
            ax.plot(R[cal,name],u[cal,name],color=f'C{j}',ls=ls,label=name)

        ax.set_xlabel('revenue, $R$')
        ax.set_title(cal)
        ax.grid(True)

    axes[0].set_ylabel('utility, $u$')
    axes[0].legend(fontsize=8)

    fig.tight_layout()
