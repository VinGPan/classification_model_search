# Data distribution
# hist + box plots + bin-wise
# image plots , text, circle, bounding box etc, random colors
# scatter with different classes with different color, different radius
import numpy as np
import matplotlib.pyplot as plt

# https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
def single_column_discrete(X, config):
    mu = np.mean(X)
    st = np.std(X)
    fig, axs = plt.subplots(3, 1, figsize=(20, 15))

    axs[0].plot(X)

    hist, edges = np.histogram(X, bins=config['bins'])
    mids = []
    for i, v in enumerate(edges[0:-1]):
        v2 = edges[i + 1]
        mids.append((v + v2) / 2)
    axs[1].bar(mids, hist)

    qs = np.quantile(X, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axs[2].plot(sorted(X), [0] * X.shape[0], '*')

    for q in qs:
        axs[2].plot([q, q], [-1, 1])

    axs[0].grid()
    axs[0].set_title(config['title'])
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel(config['ylabel'])

    axs[0].tick_params(
        axis='x',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    axs[1].set_xlabel(config['ylabel'])
    axs[1].set_ylabel('count')
    axs[1].grid()
    axs[1].text(np.min(mids), np.max(hist), r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(mu, st))

    axs[2].set_xlabel(config['ylabel'])
    axs[2].set_xticks(qs)

    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    plt.show()


def single_column_continuous(X, config):
    mu = np.mean(X)
    st = np.std(X)
    fig, axs = plt.subplots(3, 1, figsize=(20, 15))

    axs[0].plot(X)

    hist, edges = np.histogram(X, bins=config['bins'])
    mids = []
    for i, v in enumerate(edges[0:-1]):
        v2 = edges[i+1]
        mids.append((v + v2)/2)
    axs[1].bar(mids, hist)

    qs = np.quantile(X, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    axs[2].plot(sorted(X), [0] * X.shape[0], '*')

    for q in qs:
        axs[2].plot([q, q], [-1, 1])

    axs[0].grid()
    axs[0].set_title(config['title'])
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel(config['ylabel'])

    axs[0].tick_params(
        axis='x',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    axs[1].set_xlabel(config['ylabel'])
    axs[1].set_ylabel('count')
    axs[1].grid()
    axs[1].text(np.min(mids), np.max(hist), r'$\mu$={:.2f}, $\sigma$={:.2f}'.format(mu, st))

    axs[2].set_xlabel(config['ylabel'])
    axs[2].set_xticks(qs)

    plt.tick_params(
        axis='y',  # changes apply to the y-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    plt.show()


def two_columns_correl_discrete(X, config):
    pass


def two_columns_correl_continuous(X, config):
    pass


def two_columns_covar_discrete(X, config):
    pass


def two_columns_covar_continuous(X, config):
    pass


def multi_columns_correl_discrete(X, config):
    pass


def multi_columns_correl_continuous(X, config):
    pass


def multi_columns_covar_discrete(X, config):
    pass


def multi_columns_covar_continuous(X, config):
    pass


def plot(X, config):
    if isinstance(X, np.ndarray):
        print(X.shape)
        if len(X.shape) == 1:
            if X.dtype == np.dtype(float):
                single_column_continuous(X, config)
        else:
            if X.dtype == np.dtype(float):
                single_column_discrete(X, config)
            print("Not supported yet")
    else:
        print("Not supported yet")
    # is python array
    # is numpy array
    # is data frame

    # how many rows and how many columns
    # what is the type of elements


if __name__ == '__main__':
    import random

    config = {'title': 'Title',
              'ylabel': 'ylabel',
              'bins': 10,
             }

    random.seed(42)
    mn = random.randint(0, 250)
    mx = random.randint(mn, 250)
    df = mx - mn
    elems_cnt = random.randint(500, 1000)
    elems = np.array([random.random() for v in range(elems_cnt)])
    elems = (elems * df) + mn
    plot(elems, config)
    i = 0
