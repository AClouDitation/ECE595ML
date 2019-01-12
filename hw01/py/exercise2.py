import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def newfig():
    fig = plt.figure(figsize=(9,6), dpi=300)
    ax = fig.add_subplot(111)
    return fig, ax

def final_adjust(fn):
    plt.tight_layout()
    plt.savefig(fn, bbox='tight')

if __name__ == '__main__':

    # (b)   TODO: revise this
    fig, ax = newfig()
    mu, sigma = 0, 1
    X = np.random.normal(mu, sigma, 1000)
    ax.set_ylim([-3,3])
    ax.set_xlabel('sample index', fontsize = 20)
    ax.set_ylabel('sample value (x)', fontsize = 20)
    ax.grid(True)
    ax.plot(X,'r.')
    final_adjust('../pix/exercise2_b.pdf')

    # (c) 
    X = np.random.normal(mu, sigma, 1000)

    # 4 bins
    fig, ax = newfig()
    ax.hist(X, 4, density=True)
    mu, sigma = norm.fit(X)
    x_ticks = np.linspace(-5, 5, 100)
    pdf = norm.pdf(x_ticks, mu, sigma)
    ax.grid(True)
    ax.plot(x_ticks, pdf, 'k', linewidth=2)
    ax.set_title("Fit value: mu = %.2f, sigma = %.2f"%(mu, sigma))
    ax.set_xlabel('sample value', fontsize = 20)
    ax.set_ylabel('frequency', fontsize = 20)
    final_adjust('../pix/exercise2_c1.pdf')

    # 1000 bins
    fig, ax = newfig()
    ax.hist(X, 1000, density=True)
    mu, sigma = norm.fit(X)
    x_ticks = np.linspace(-5, 5, 100)
    pdf = norm.pdf(x_ticks, mu, sigma)
    ax.grid(True)
    ax.plot(x_ticks, pdf, 'k', linewidth=2)
    ax.set_title("Fit value: mu = %.2f, sigma = %.2f"%(mu, sigma))
    ax.set_xlabel('sample value', fontsize = 20)
    ax.set_ylabel('frequency', fontsize = 20)
    final_adjust('../pix/exercise2_c2.pdf')
    
    # (d)
    n = 1000
    J_h = np.zeros(200)
    for m in range(1,201):
        h = (max(X) - min(X))/m
        p = np.array([((X >= min(X) + h*x) & (X < min(X) + h*(x+1))).sum()/n for x in range(m)])
        J_h[m-1] = 2 / h / (n-1) - (n+1) / h /(n-1) * sum(p**2)

    fig, ax = newfig()
    ax.grid(True)
    ax.plot([x for x in range(1,201)], J_h)
    ax.set_xlabel('# of bins', fontsize = 20)
    ax.set_ylabel('CVER', fontsize = 20)
    final_adjust('../pix/exercise2_d1.pdf')

    m_star = J_h.argmin()
    
    fig, ax = newfig()
    ax.hist(X, m_star, density=True)
    mu, sigma = norm.fit(X)
    x_ticks = np.linspace(-5, 5, 100)
    pdf = norm.pdf(x_ticks, mu, sigma)
    ax.grid(True)
    ax.plot(x_ticks, pdf, 'k', linewidth=2)
    ax.set_title("Fit value: mu = %.2f, sigma = %.2f"%(mu, sigma))
    ax.set_xlabel('sample value', fontsize = 20)
    ax.set_ylabel('frequency', fontsize = 20)
    final_adjust('../pix/exercise2_d2.pdf')
