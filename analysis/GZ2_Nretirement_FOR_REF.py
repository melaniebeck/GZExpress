import numpy as np
import pandas as pd
from astropy.table import Table
import pdb
import sys
from simulation import Simulation

truth = {'SMOOTH':1, 'NOT':0, 'UNKNOWN':-1}

def get_column_names(vote_fraction_type):

	if vote_fraction_type == 'raw':           suffix = 'fraction'
	elif vote_fraction_type == 'weighted':    suffix = 'weighted_fraction'
	elif vote_fraction_type == 'debiased':    suffix = 'debiased'
	else: 
		print "%s is not a GZ2 vote option"%type
		sys.exit()

	smooth = 't01_smooth_or_features_a01_smooth_%s'%suffix
	featured = 't01_smooth_or_features_a02_features_or_disk_%s'%suffix
	star = 't01_smooth_or_features_a03_star_or_artifact_%s'%suffix

	return smooth, featured, star


def compute_label(smooth, featured, star, threshold=0.5):

	combo = featured + star
	#pdb.set_trace()

	if combo >= threshold:
		return truth['NOT']
	elif combo < threshold:
		return truth['SMOOTH']
	else:
		return truth['UNKNOWN']


def compute_swap_label(value):
	if value >= 0.99:
		return 0
	elif value <= 0.004:
		return 1
	else: 
		return -1


def plot_results():
	# Where did I check these??? Damn you, Melanie!
	SWAP_only_total_votes = 2298772
	SWAP_only_accuracy = .957
	SWAP_only_subjects_retired = 226124

	GZX_total_votes = 932017
	GZX_accuracy = .935
	GZX_subjects_retired = 210543

	GZ2_vary_retirement = Table.read("GZ2_vary_retirement_criterion.csv")
	N = GZ2_vary_retirement['votes_per_gal']
	total_votes = GZ2_vary_retirement['total_votes']
	acc = GZ2_vary_retirement['accuracy']

	import matplotlib.pyplot as plt 

	fig = plt.figure(figsize=(13,8))
	ax = fig.add_subplot(111)

	ax.scatter(N, acc,  marker='^', s=100, color='red')
	ax.scatter(10, SWAP_only_accuracy,  marker='^', s=100, color='red')
	ax.scatter(5, GZX_accuracy,  marker='^', s=100, color='red')

	gz2_retirement = [str(n) for n in N]

	ax.set_xticks(np.arange(5,40,5))
	ax.set_xticklabels(['GZX', 'SWAP']+gz2_retirement)
	#plt.scatter(N, total_votes/float(SWAP_only_total_votes))
	ax.set_ylabel("Accuracy", color='red')
	ax.tick_params('y', colors='red')
	ax.set_xlabel("                                          GZ2 retirement limit")


	ax2 = ax.twinx()
	ax2.bar(N, N*SWAP_only_subjects_retired/1e6, width=2, color='gray', alpha=0.4)
	ax2.bar(10, SWAP_only_total_votes/1e6, width=2, color='gray', alpha=0.4)
	ax2.bar(5, GZX_total_votes/1e6, width=2, color='gray', alpha=0.4)
	ax2.set_ylabel("Human Effort [1e6]")

	plt.savefig('retirement_limites.png')
	plt.show()

	pdb.set_trace()



sim = Simulation(config="update_sup_PLPD5_p5_flipfeature2b_norandom2.config", 
				 directory="S_PLPD5_p5_ff_norand/", 
				 variety='feat_or_not')


retired = sim.fetchCatalog(sim.retiredFileList[-1])
retired = retired.to_pandas()
retired['name'] = retired['zooid']
retired['swap_label'] = retired.apply(lambda x: compute_swap_label(x['P']), axis=1)
	
#plot_results()


votesall = pd.read_csv('multi-threshold_GZ2_labels.csv')


N = np.arange(15, 40, 5)
N = np.insert(N, 0, 9, axis=0)
print N

swap_only_acc = []
full_acc = []
total_votes = []

for n in N: 

	try: 
		votesfewer = pd.read_table('asset_agg{}votes_task1.tsv'.format(n), 
								   delimiter='\t', header=None, 
								   names=['asset_id', 'name', 'smooth_count', 'feature_count', 'star_count'])
	except:
		votesfewer =pd.read_table('asset_agg{}votes.tsv'.format(n), delimiter='\t')
	
	votesfewer = votesfewer[votesfewer.name < 7000000000000000000]


	# How many votes were there actually (some subjects have fewer than 30)
	votesfewer['total_votes'] = votesfewer.smooth_count + votesfewer.feature_count + votesfewer.star_count


	# Turn the counts into "vote fractions"
	votesfewer['smooth_fraction'] = votesfewer.smooth_count / votesfewer['total_votes']
	votesfewer['feature_fraction'] = votesfewer.feature_count / votesfewer['total_votes']
	votesfewer['star_fraction'] = votesfewer.star_count / votesfewer['total_votes']

	# Compute labels
	votesfewer['GZ2_raw{}label_0.5'.format(n)] = votesfewer.apply(lambda x: compute_label(x['smooth_fraction'], 
																		  	   x['feature_fraction'], 
																		  	   x['star_fraction']), axis=1)

	# Save this to file for later exploration
	votesfewer.to_csv("asset_agg{}votes_task1.csv".format(n))

	# select only those that were also retired by SWAP
	result1 = pd.merge(votesfewer, retired, on='name')
	result1 = pd.merge(votesall, result1, on='asset_id')
	#clean_result1 = result1.loc[result1['GZ2_raw_label_0.5'] >= 0]
	clean_result1 = result1

	# filter out those subjects which never had published GZ2 vote fractions
	result2 = pd.merge(votesall, votesfewer, on='asset_id')
	#clean_result2 = result2.loc[result2['GZ2_raw_label_0.5'] >= 0]
	clean_result2 = result2


	swap_only_acc.append(np.sum(clean_result1['GZ2_raw{}label_0.5'.format(n)] == 
						   		clean_result1['GZ2_raw_label_0.5'])/float(len(clean_result1)))

	full_acc.append(np.sum(clean_result2['GZ2_raw{}label_0.5'.format(n)] == 
						   clean_result2['GZ2_raw_label_0.5'])/float(len(clean_result2)))

	#pdb.set_trace()


"Finished processing all N vote aggregations."
pdb.set_trace()

tt = Table(data=[N, swap_only_acc, full_acc], 
		   names=("votes_per_gal", "swap_sample_acc", "gz2_sample_acc"))

tt.write("GZ2_vary_retirement_criterion.csv")

pdb.set_trace()