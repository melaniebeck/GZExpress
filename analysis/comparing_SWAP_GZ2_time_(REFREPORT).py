import numpy as np
import pandas as pd
import pdb, sys
import cPickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/home/oxymoronic/research/GZExpress/analysis/')
from simulation import Simulation


import matplotlib as mpl
mpl.rcParams.update({'font.size': 24, 
							'font.family': 'STIXGeneral', 
							'mathtext.fontset': 'stix',
							'xtick.labelsize':18,
							'ytick.labelsize':18,
							'xtick.major.width':2,
							'ytick.major.width':2,
							'axes.linewidth':2,
							'lines.linewidth':3,
							'legend.fontsize':18})


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


def run_monte_carlo(gz2, N, Nsamples=100):

	accuracy = {}
	total_votes = {}

	for n in N:
		print "computing MonteCarlo for N={} votes".format(n)

		accuracy["{}agg".format(n)] = []
		total_votes["{}agg".format(n)] = []

		try: 
			gz2['GZ2_raw{}label_0.5'.format(n)]
		except:
			Nagg = pd.read_csv("asset_agg{}votes_task1.csv".format(n), 
							   usecols=["name","GZ2_raw{}label_0.5".format(n)])
			gz2 = gz2.merge(Nagg, on='name')

		for i in range(Nsamples): 
			# Choose a random (with replacement) set of indices from the full
			# GZ2 list of the same size as the SWAP-retired sample
			random_idx = np.random.choice(np.array(len(gz2)), np.array(len(SWAP_retired)), 
										  replace=False)

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

	return accuracy, total_votes

plot = True

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


# Join SWAP catalogs with GZ2 vote fraction catalog
SWAP_retired['name'] = SWAP_retired['zooid']
SWAP_retired = SWAP_retired.merge(gz2, on='name').dropna()

SWAP_never_retired['name'] = SWAP_never_retired['zooid']
SWAP_never_retired = SWAP_never_retired.merge(gz2, on='name').dropna()

smoothFrac = 't01_smooth_or_features_a01_smooth_fraction'
featFrac = 't01_smooth_or_features_a02_features_or_disk_fraction'


N = np.arange(10, 40, 5)
N = np.insert(N, 0, 9)

# If I've already run the random trials -- open them up
# otherwise, run them!
#"""
try:
	with open('GZ2_Nagg_accuracy_for_random_SWAP_samples.pickle', 'rb') as F:
		accuracy = cPickle.load(F)	

	with open('GZ2_Nagg_total_votes_for_random_SWAP_samples.pickle', 'rb') as F:
		total_votes = cPickle.load(F)
except:
	run_monte_carlo()
#"""

#accuracy, total_votes = run_monte_carlo(gz2, N)

#  That data are just the raw numbers: take some simple stats
acc_mean, acc_std = [], []
votes_mean, votes_std = [], []

for n in N:
	key = "{}agg".format(n)

	acc_mean.append(np.mean(accuracy[key]))
	acc_std.append(np.std(accuracy[key]))

	votes_mean.append(np.mean(total_votes[key]))
	votes_std.append(np.std(total_votes[key]))

# Get SWAP simulation values for accuracy and total votes
SWAP_acc = compute_accuracy(SWAP_retired, 'swap_label', 'GZ2_raw_label_0.5')
SWAP_votes = np.sum(SWAP_retired['Nclass'])


GZ2_full_votes = 14144941.

if plot:
	# if using 10 instead of 9:
	N = N[1:]

	fig = plt.figure(figsize=(15, 10))
	gs = gridspec.GridSpec(6, 4)

	votes_mean_frac = np.array(votes_mean[1:])/float(votes_mean[-1])

	################# PLOT ACCURACY AND TOTAL VOTES ####################
	ax1 = plt.subplot(gs[3:, :])
	ax2 = ax1.twinx()

	#yerr=np.array(votes_std[1:])/1e6, 
	ax1.bar(N, votes_mean_frac, width=2, color='gray', alpha=0.4)
	ax1.bar(5, float(SWAP_votes)/votes_mean[-1], width=2, color='gray', alpha=1.)
	
	ax1.yaxis.tick_right()
	ax1.yaxis.set_label_position("right")
	ax1.set_ylabel(r"Fraction of votes at $N=35$")

	NN = np.insert(N, 0, 5)
	ax1.set_xticks(NN)
	ax1.set_xticklabels(['SWAP']+[str(n) for n in N])
	ax1.set_xlabel("                   GZ2 retirement at N votes")


	ax2.errorbar(N, acc_mean[1:], yerr=acc_std[1:], fmt='o', markersize=10, 
				 capthick=2, color='red')
	ax2.errorbar(5, SWAP_acc, fmt='o', markersize=10, capthick=2, color='red')

	ax2.yaxis.tick_left()
	ax2.yaxis.set_label_position("left")
	ax2.set_ylabel("Accuracy", color='red')
	ax2.tick_params('y', colors='red')

	################# PLOT GZ2 SMOOTH FRACTION ####################
	ax = plt.subplot(gs[0:3,:2])
	ax.hist(gz2[smoothFrac].dropna().values, bins=50, histtype='stepfilled', 
			color='orange', alpha=0.25, edgecolor='grey', 
			label='All GZ2')
	ax.hist(SWAP_retired[smoothFrac], bins=50, histtype='step', lw=2,  alpha=0.8,
			label="SWAP retired")
	ax.hist(SWAP_never_retired[smoothFrac], bins=50, histtype='step', color='red',
			lw=2,  alpha=0.8,
			label="SWAP not (yet) retired")

	ax.legend(loc='upper left', frameon=False)
	ax.set_xlabel("$f_{\mathrm{smooth}}$")
	ax.set_ylabel("Counts")

	################# PLOT VOTES AT SUBJECT RETIREMENT ####################
	ax = plt.subplot(gs[0:3, 2:])
	bins = np.arange(10, 80, 2)
	ax.hist(gz2['total_classifications'].dropna().values, normed=True, bins=bins, 
	        histtype='stepfilled', alpha=0.25, color='orange', edgecolor='grey', 
	        label=None)
	ax.hist(SWAP_retired['total_classifications'], normed=True, bins=bins,
	        histtype='step', lw=2, alpha=0.8,
	        label=None)
	ax.hist(SWAP_never_retired['total_classifications'], normed=True, bins=bins, 
	        histtype='step', color='red', lw=2, alpha=0.8,
	        label=None)

	bins = np.arange(0, 70, 2)
	ax.hist(SWAP_retired['Nclass'], normed=True, bins=bins, histtype='step', 
			lw=2, ls='--', color='steelblue', alpha=0.8, 
			label="Simulation: SWAP retired")
	#ax.hist(SWAP_never_retired['Nclass'], normed=True, bins=bins, histtype='step', 
	#		lw=2, ls='--', color='green', alpha=0.8, 
	#		label="Simulation: SWAP not (yet) retired")

	ax.legend(loc='upper left', frameon=False)
	ax.set_xlabel("Votes at retirement")
	ax.set_ylabel("Normalized units")
	ax.set_ylim(0, .13)

	gs.tight_layout(fig)
	plt.savefig("SWAP_vs_GZ2_retirement.pdf")
	#plt.show()
	plt.close()

pdb.set_trace()


### --------------------------------------------------------------
### RUN SOME ADDITIONAL CHECKS 
### --------------------------------------------------------------

"""
Claudia wants me to look a the ratio of GZ2 / SWAPretired +SWAPnotretired
and make sure it's constant as a function f_smooth (random sample)
"""

swap = pd.concat([SWAP_retired, SWAP_never_retired])

fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(211)
sss, bins, _ = ax.hist(swap[smoothFrac], bins=50, histtype='stepfilled', 
					   normed=True, alpha=0.5, 
					   label='swap')
ggg, bins, _ = ax.hist(gz2[smoothFrac].dropna().values, bins=50, histtype='stepfilled',
					   normed=True, alpha=0.5, 
					   label='gz2')
ax.legend(loc='upper left', frameon=False)
#ax.set_xlabel('f_smooth')

ax = fig.add_subplot(212)
ax.scatter(bins[:-1]+0.01, ggg-sss)
ax.set_xlabel('f_smooth')
ax.set_ylabel('gz2 - swap')

plt.savefig("swap_check_random_sample1.png")
#plt.show()
plt.close()


fig = plt.figure(figsize=(10,15))
ax = fig.add_subplot(211)
sss, bins, _ = ax.hist(swap['total_classifications'], bins=50, histtype='stepfilled', 
					   normed=True, alpha=0.5, 
					   label='swap')
ggg, bins, _ = ax.hist(gz2['total_classifications'].dropna().values, bins=50, histtype='stepfilled',
					   normed=True, alpha=0.5, 
					   label='gz2')
ax.legend(loc='upper left', frameon=False)

ax = fig.add_subplot(212)
ax.scatter(bins[:-1]+0.01, ggg-sss)
ax.set_xlabel('total_classifications')
ax.set_ylabel('gz2 - swap')

plt.savefig("swap_check_random_sample2.png")
#plt.show()
plt.close()

fig = plt.figure(figsize=(8,8))
ax= fig.add_subplot(111)

## The factor in front is due to the fact that I select only those votes
# made by volunteers who have seen at least one of the gold-standard galaxies
# This reduces the total pool of votes by ~10%
ax.scatter(.897*SWAP_never_retired['total_classifications'], 
		   SWAP_never_retired['Nclass'],
		   marker='.', alpha=0.25)

ax.plot([5,50],[5, 50], 'k-')

ax.set_xlabel("GZ2 votes at retirement")
ax.set_ylabel("SWAP votes at end of simulation")
ax.set_title("SWAP not retired: Vote distributions")

plt.savefig("swap_never_retired_votes.png")
plt.show()

pdb.set_trace()

