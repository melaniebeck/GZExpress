import numpy as np
import pandas as pd
import pdb, sys
import cPickle

import matplotlib.pyplot as plt

sys.path.insert(0, '/home/oxymoronic/research/GZExpress/analysis/')
from simulation import Simulation


def compute_swap_label(value):
	if value >= 0.99:
		return 0
	elif value <= 0.004:
		return 1
	else: 
		return -1

def compute_accuracy(sample, label1, label2):
	mask = sample[label1] == sample[label2]
	accuracy = np.sum(mask)/float(len(sample))
	return accuracy

plot = False

### ---------------------------------------------------------------------------
### READ IN ALL THE SWAP DATA / CATALOGS
### ---------------------------------------------------------------------------
sim = Simulation(config="update_sup_PLPD5_p5_flipfeature2b_norandom2.config", 
				 directory="S_PLPD5_p5_ff_norand/", 
				 variety='feat_or_not')

candidates = sim.fetchCatalog(sim.fetchFileList(kind='candidate')[-1])
#rejected = sim.fetchCatalog(sim.fetchFileList(kind='rejected')[-1])
#detected = sim.fetchCatalog(sim.fetchFileList(kind='detected')[-1])
SWAP_retired = sim.fetchCatalog(sim.fetchFileList(kind='retired')[-1])
SWAP_retired['swap_label'] = SWAP_retired.apply(lambda x: compute_swap_label(x['P']), axis=1)

# Let's look at the SWAP-never-retired sample
# First find subjects the candidates and retired catalogs have in common
common = candidates.merge(SWAP_retired, on=['zooid'])

# Now select out those subjects in candidates which are not part of the common group
SWAP_never_retired = candidates[(~candidates.zooid.isin(common.zooid))]


### ---------------------------------------------------------------------------
### READ IN ALL THE GZ2 LABELS / CATALOGS / ETC
### ---------------------------------------------------------------------------
gz2_votefracs = pd.read_csv("GZ2assets_vote_fractions.csv")

gz2_labels = pd.read_csv("multi-threshold_GZ2_labels.csv")
gz2_labels['name'] = gz2_labels['SDSS_id']

gz2_Nagg = pd.read_csv("asset_agg9votes_task1.csv")

gz2 = gz2_votefracs.merge(gz2_labels, on='name')

# Resulting table has GZ2 vote fractions, multi-threshold labels, 
# and N aggregation labels for all 295K subjects
gz2 = gz2.merge(gz2_Nagg, on='name')


### ---------------------------------------------------------------------------
### IS SWAP RETIRING THE "EASIEST" SUBJECTS?  CHECK VOTE FRAC DISTS
### ---------------------------------------------------------------------------
# Join SWAP catalogs with GZ2 vote fraction catalog
SWAP_retired['name'] = SWAP_retired['zooid']
SWAP_retired = SWAP_retired.merge(gz2, on='name').dropna()

SWAP_never_retired['name'] = SWAP_never_retired['zooid']
SWAP_never_retired = SWAP_never_retired.merge(gz2, on='name').dropna()

smoothFrac = 't01_smooth_or_features_a01_smooth_fraction'
featFrac = 't01_smooth_or_features_a02_features_or_disk_fraction'

if plot:
	fig = plt.figure(figsize=(10, 7))

	# Plot vote fraction distributions for SWAP samples against the full GZ2
	ax = fig.add_subplot(121)
	ax.hist(gz2[smoothFrac].dropna().values, bins=50, histtype='stepfilled', 
			color='orange', alpha=0.25, edgecolor='orange', 
			label='All GZ2')
	ax.hist(SWAP_retired[smoothFrac], bins=50, histtype='step', lw=2, 
			label="SWAP retired")
	ax.hist(SWAP_never_retired[smoothFrac], bins=50, histtype='step', color='green',
			lw=2, ls='--',
			label="SWAP not (yet) retired")
	ax.legend(loc='upper left', frameon=False)
	ax.set_xlabel("$f_{\mathrm{smooth}}$", fontsize=20)

	"""
	ax = fig.add_subplot(222)
	ax.hist(SWAP_retired[featFrac], bins=50, histtype='step', label="SWAP-retired")
	ax.hist(SWAP_never_retired[featFrac], bins=50, histtype='step', label="SWAP-not-retired")
	ax.hist(gz2[featFrac].dropna().values, bins=50,histtype='step', color='orange', label='All GZ2')
	ax.legend(loc='upper right', frameon=False)
	ax.set_xlabel("$f_{\mathrm{featured}}$")
	"""

	# Plot votes till retirement for SWAP samples against full GZ2
	ax = fig.add_subplot(122)
	bins = np.arange(10, 80, 2)
	ax.hist(gz2['total_classifications'].dropna().values, normed=True, bins=bins, 
	        histtype='stepfilled', alpha=0.25, color='orange', edgecolor='orange', 
	        label="All GZ2")
	ax.hist(SWAP_retired['total_classifications'], normed=True, bins=bins,
	        histtype='step', lw=2, 
	        label="SWAP retired")
	ax.hist(SWAP_never_retired['total_classifications'], normed=True, bins=bins, 
	        histtype='step', color='green', lw=2, ls='--',
	        label="SWAP not (yet) retired")
	#ax.legend(loc='upper left', frameon=False)
	ax.set_xlabel("GZ2 votes until retirement", fontsize=16)
	ax.set_ylim(0, .12)

	plt.savefig("SWAP-GZ2_votefracs_voteclicks.png")
	plt.show()

	pdb.set_trace()


accuracy = {}
total_votes = {}

Nsamples = 100
N = np.arange(15, 40, 5)
N = np.insert(N, 0, 9)

# If I've already run the "monte carlo", just open the accuracy pickle
try:
	with open('GZ2_Nagg_accuracy_for_random_SWAP_samples.pickle', 'rb') as F:
		accuracy = cPickle.load(F)	

	with open('GZ2_Nagg_total_votes_for_random_SWAP_samples.pickle', 'rb') as F:
		total_votes = cPickle.load(F)

except:
	for n in N:
		print "computing MonteCarlo for N={} votes".format(n)

		accuracy["{}agg".format(n)] = []
		total_votes["{}agg".format(n)] = []

		try: 
			gz2['GZ2_raw{}label_0.5'.format(n)]
		except:
			Nagg = pd.read_csv("asset_agg{}votes_task1.csv".format(n), usecols=[2,10])
			gz2 = gz2.merge(Nagg, on='name')

		for i in range(Nsamples): 
			# Choose a random (with replacement) set of indices from the full
			# GZ2 list of the same size as the SWAP-retired sample
			random_idx = np.random.choice(np.array(len(gz2)), np.array(len(SWAP_retired)))

			# From the GZ2 catalog, isolate that subjects belonging to those indices
			random_sample = gz2.loc[random_idx].reset_index()

			accuracy["{}agg".format(n)].append(compute_accuracy(random_sample, 
										 						'GZ2_raw{}label_0.5'.format(n), 
										 						'GZ2_raw_label_0.5'))

			fewer_than_N = random_sample.loc[np.where(random_sample['total_classifications'] < n)]
			at_least_N = len(random_sample)-len(fewer_than_N)

			votes = at_least_N*n + np.sum(fewer_than_N['total_classifications'])
			total_votes["{}agg".format(n)].append(votes)

with open('GZ2_Nagg_accuracy_for_random_SWAP_samples.pickle', 'wb') as F:
	cPickle.dump(accuracy, F)	

with open('GZ2_Nagg_total_votes_for_random_SWAP_samples.pickle', 'wb') as F:
	cPickle.dump(total_votes, F)	

"""
for k, v in total_votes.iteritems():
	print k, np.mean(v), np.std(v)

for k, v in accuracy.iteritems():
	print k, np.mean(v), np.std(v)
"""

acc_mean, acc_std = [], []
votes_mean, votes_std = [], []

for n in N:
	key = "{}agg".format(n)

	acc_mean.append(np.mean(accuracy[key]))
	acc_std.append(np.std(accuracy[key]))

	votes_mean.append(np.mean(total_votes[key]))
	votes_std.append(np.std(total_votes[key]))


SWAP_acc = compute_accuracy(SWAP_retired, 'swap_label', 'GZ2_raw_label_0.5')
SWAP_votes = np.sum(SWAP_retired['Nclass'])

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

ax.errorbar(N, acc_mean, yerr=acc_std, fmt='o', capthick=2, color='red')
ax.errorbar(5, SWAP_acc, fmt='o', capthick=2, color='red')


NN = np.insert(N, 0, 5)
ax.set_xticks(NN)
ax.set_xticklabels(['SWAP']+[str(n) for n in N])
ax.set_ylabel("Accuracy", color='red')
ax.tick_params('y', colors='red')
ax.set_xlabel("                   GZ2 N votes till retirement")


ax2 = ax.twinx()
ax2.bar(N, np.array(votes_mean)/1e6, yerr=np.array(votes_std)/1e6, width=2, 
		color='gray', alpha=0.4)
ax2.bar(5, float(SWAP_votes)/1e6, width=2, color='gray', alpha=0.4)
ax2.set_ylabel("Total classifications [1e6]")

plt.savefig('SWAP_vs_GZ2_Nagg_retirement.png')
plt.show()

pdb.set_trace()
