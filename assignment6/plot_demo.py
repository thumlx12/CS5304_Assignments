from matplotlib import pyplot as plt
import numpy as np

mu, sigma, beta, n = 0, 1, 10, 10000
alpha = 0.2

x = np.random.normal(mu, sigma, n)
y = np.random.exponential(beta, n) - np.random.normal(mu, sigma, n)

plt.figure()
plt.title('Scatter Example')
plt.xlabel('Normal (mu={}, sigma={})'.format(mu, sigma))
plt.ylabel('Exponential (beta={}) - Normal (mu={}, sigma={})'.format(beta, mu, sigma))
plt.scatter(x, y)
plt.tight_layout()
plt.savefig('scatter.png')
plt.close()

plt.figure()
plt.title('Scatter Example (alpha={})'.format(alpha))
plt.xlabel('Normal (mu={}, sigma={})'.format(mu, sigma))
plt.ylabel('Exponential (beta={}) - Normal (mu={}, sigma={})'.format(beta, mu, sigma))
plt.scatter(x, y, alpha=alpha)
plt.tight_layout()
plt.savefig('scatter_alpha.png')
plt.close()

plt.figure()
plt.title('Heatmap Example')
plt.xlabel('Normal (mu={}, sigma={})'.format(mu, sigma))
plt.ylabel('Exponential (beta={}) - Normal (mu={}, sigma={})'.format(beta, mu, sigma))
plt.hist2d(x, y, (50, 50), cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
plt.savefig('heatmap.png')
plt.close()