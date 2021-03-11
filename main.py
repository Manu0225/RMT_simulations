from tqdm import tqdm
import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def compute_Sigma_n(n, c, sigma, ls_theta, n_batch):
	N = int(np.floor(n * c))
	u = np.random.uniform(-1, 1, size=(N, 1))
	u = u / la.norm(u)
	R_N = sigma ** 2 * (np.eye(N)[None, :, :] + ls_theta[:, None, None] * (u @ u.T)[None, :, :])
	assert R_N.shape == (len(ls_theta), N, N)

	X_N = np.random.normal(size=(n_batch, N, n))
	assert X_N.shape == (n_batch, N, n), X_N.shape
	Sigma_n = 1 / np.sqrt(n) * np.array([np.einsum('ik,bkj->bij', sqrtm(R_N[i]), X_N) for i in range(len(ls_theta))])
	# Sigma_n = 1 / np.sqrt(n) * np.array([sqrtm(R_N[i]) @ X_N[0] for i in range(len(ls_theta))])[:,None]

	# np.random.normal((n_batch, ls_theta,))

	assert Sigma_n.shape == (len(ls_theta), n_batch, N, n)
	return Sigma_n


def batch_largest_eigval(Sigma_n):
	return la.norm(Sigma_n, ord=2, axis=(-2, -1)) ** 2


def main1():
	n = int(400)
	c = .36
	sigma = 1
	ls_theta = np.linspace(1e-3, 2)
	# ls_theta = np.array([0.2, 0.7])
	n_batch = 10
	Sigma_n = compute_Sigma_n(n, c, sigma, ls_theta, n_batch)

	eigvals = batch_largest_eigval(Sigma_n)
	avg_eigvals = np.mean(eigvals, axis=-1)
	assert avg_eigvals.shape == (len(ls_theta),)
	std_eigvals = np.std(eigvals, axis=-1)

	fig = plt.figure(figsize=(10, 7))

	limit_1 = sigma ** 2 * (1 + np.sqrt(c)) ** 2
	ls_limit_2 = sigma ** 2 * (1 + ls_theta) * (1 + c / ls_theta)
	theoretica_limit = (ls_theta <= np.sqrt(c)) * limit_1 + (ls_theta > np.sqrt(c)) * ls_limit_2
	plt.plot(ls_theta, avg_eigvals,
			 ".-",
			 label=f"valeur propre maximale (moyennée sur {n_batch} itérations)",
			 )
	plt.fill_between(ls_theta, avg_eigvals - std_eigvals, avg_eigvals + std_eigvals,
					 label=f"écart-type sur {n_batch} itérations",
					 alpha=.3)
	plt.plot(ls_theta,
			 theoretica_limit,
			 label=f"limite théorique")
	# plt.plot

	plt.ylim(np.min(theoretica_limit)-.5, np.max(theoretica_limit)+.5)

	plt.axvline(x=np.sqrt(c),
				label=r'$\sqrt{c}$',
				color="grey",
				linestyle="--",
				alpha=.6)

	plt.legend()
	plt.xlabel(r"$\theta$",
			   fontsize="x-large"
			   )
	plt.ylabel(r'$\lambda_{max}$',
			   fontsize="x-large"
			   )

	fig.tight_layout()

	plt.savefig(f"D:/Users/emmanuel/OneDrive/M2/boulot/Random Matrix Theory/img/fig_c={c}.png",
				# bbox_inches='tight',
				pad_inches=0)
	# plt.show()


def main2():
	ls_n = range(50, 700, 60)  # int(300)
	c = .36
	sigma = 1
	ls_theta = np.linspace(1e-1, 2, num=5)
	# ls_theta = np.array([0.2, 0.7])
	n_batch = 10
	eigvals = np.empty((len(ls_n), len(ls_theta), n_batch))
	for i, n in enumerate(tqdm(ls_n)):
		eigvals[i] = batch_largest_eigval(compute_Sigma_n(n, c, sigma, ls_theta, n_batch))
	avg_eigvals = np.mean(eigvals, axis=-1)
	assert avg_eigvals.shape == (len(ls_n), len(ls_theta))
	std_eigvals = np.std(eigvals, axis=-1)

	fig = plt.figure(figsize=(10, 7))
	line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
				   '#17becf']  # ['blue', 'orange', 'b', 'c', 'm', 'y', 'k']#, ...]
	for i, theta in enumerate(ls_theta):
		plt.plot(ls_n,
				 avg_eigvals[:, i],
				 ".-",
				 label=f"θ = {theta:.1f}",
				 c=line_colors[i],

				 )
		plt.fill_between(ls_n, (avg_eigvals - std_eigvals)[:, i], (avg_eigvals + std_eigvals)[:, i],
						 label=f"écart-type ({n_batch} itérations)",
						 color=line_colors[i],
						 alpha=.3)

		limit_1 = sigma ** 2 * (1 + np.sqrt(c)) ** 2
		limit_2 = sigma ** 2 * (1 + theta) * (1 + c / theta)

		true_limit = limit_1 if theta <= np.sqrt(c) else limit_2
		plt.plot(ls_n,
				 np.ones(len(ls_n)) * true_limit,
				 c=line_colors[i],
				 label=f"limite théorique (θ={theta:.1f})",
				 alpha=.7,
				 ls="--"
				 )
	plt.legend(
		# handles=[p1, p2],
		# 	   title='title',
		bbox_to_anchor=(0, 1.02, 1, .5),  # 1.05, 1),
		fontsize="medium",
		loc='lower left',
		mode="expand",
		ncol=3
		# prop=fontP
	)
	plt.xlabel("$n$",
			   fontsize="x-large"
			   )
	plt.ylabel(r'$\lambda_{max}$',
			   fontsize="x-large"
			   )

	plt.ylim(2, 4)

	fig.tight_layout()

	plt.savefig("D:/Users/emmanuel/OneDrive/M2/boulot/Random Matrix Theory/img/fig.png",
				# bbox_inches='tight',
				pad_inches=0)


# plt.show()


main1()
# main2()
