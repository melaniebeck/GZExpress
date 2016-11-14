from __future__ import division

from simulation import Simulation
from astropy.table import Table, join, vstack
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib
import pdb, sys
from datetime import *
import cPickle
from GZX_SWAP_evaluation import generate_SWAP_eval_report, \
								calculate_confusion_matrix, \
								GZ2_label_SMOOTH_NOT

from find_indices import find_indices


matplotlib.rcParams.update({'font.size': 20, 
							'font.family': 'STIXGeneral', 
							'mathtext.fontset': 'stix',
							'xtick.labelsize':24,
							'ytick.labelsize':24,
							'xtick.major.width':2,
							'ytick.major.width':2,
							'axes.linewidth':2,
							'lines.linewidth':3,
							'legend.fontsize':20})


def collect_probabilities(bureau):
	bureau_stuff = {'PLarray':[],
					'PDarray':[],
					'contributions':[],
					'skills':[],
					'Ntraining':[],
					'Ntest':[],
					'Ntotal':[]}

	for ID in bureau.list():
		agent = bureau.member[ID]
		bureau_stuff['PLarray'].append(agent.PL)
		bureau_stuff['PDarray'].append(agent.PD)
		bureau_stuff['contributions'].append(agent.contribution)
		bureau_stuff['skills'].append(agent.skill)
		bureau_stuff['Ntraining'].append(agent.NT)  
		bureau_stuff['Ntotal'].append(agent.N)  

	for k,v in bureau_stuff.iteritems():
		bureau_stuff[k] = np.array(bureau_stuff[k])

	bureau_stuff['Ntest'] = bureau_stuff['Ntotal'] - bureau_stuff['Ntraining']

	return bureau_stuff


###############################################################################
#			BASELINE RUN
###############################################################################
def plot_GZX_baseline(GZX_baseline_run, GZX_baseline_eval, gz2_retired):

	days = np.arange(GZX_baseline_run.days)


	fig = plt.figure(figsize=(12,8))
	ax = fig.add_subplot(111)
	ax.grid(linestyle='dotted',linewidth=.5)

	GZX_baseline_run.plot_cumulative_retired_subjects(ax, plot_combined=True)
	ax.plot(days, gz2_retired, color='midnightblue', lw=3, label='GZ2')

	for tl in ax.get_yticklabels():
		tl.set_color('steelblue')

	ax.set_ylabel(r'Cumulative retired subjects $[10^3]$', color='steelblue')

	ax.legend(loc=(0.01, .67), frameon=False)


	# OVERLAID FIGURE -- EVAL METRICS
	# -----------------------------------------------------
	ax2 = ax.twinx()

	ax2.plot(days, GZX_baseline_eval['accuracy'], c='black', label='Accuracy')
	ax2.plot(days, GZX_baseline_eval['recall'], c='dimgrey', ls='-.', label='Completeness')
	ax2.plot(days, GZX_baseline_eval['precision'], c='grey', ls='--', label='Purity')

	for tl in ax2.get_yticklabels():
		tl.set_color('dimgrey')

	ax2.legend(loc=(0.01, 0.47), frameon=False)

	ax2.set_xlim(0, GZX_baseline_run.days-1)
	ax2.set_ylim(0., 1.0)

	ax2.set_xlabel('Days in GZ2 project')
	ax2.set_ylabel('Proportion',rotation=270, labelpad=20, color='dimgrey')

	plt.savefig('GZX_eval_and_retirement_baseline_4paper.png')
	plt.show()
	exit()


###############################################################################
#			EVALUATION METRICS: SPREAD DUE TO MULTI SIMS
###############################################################################

def plot_GZX_evaluation_spread(num_days, GZX_low_run, GZX_baseline_run, 
							   GZX_high_run, GZX_p35_run, outfile, ax=None):
	
	single_figure = False
	days = np.arange(num_days)

	if not ax:
		print "create the damn axes..."
		fig = plt.figure(figsize=(10,8))
		ax = fig.add_subplot(111)
		single_figure = True
	
	ax.grid(linestyle='dotted',linewidth=.5)

	min_acc, max_acc = [], []
	for la, ma, ha, p35a in zip(GZX_low_run['accuracy'], GZX_baseline_run['accuracy'],
						  		GZX_high_run['accuracy'], GZX_p35_run['accuracy']):
		min_acc.append(np.min([la, ma, ha, p35a]))
		max_acc.append(np.max([la, ma, ha, p35a]))

	min_rec, max_rec = [], []
	for lr, mr, hr, p35r in zip(GZX_low_run['recall'], GZX_baseline_run['recall'],
						  		GZX_high_run['recall'], GZX_p35_run['recall']):
		min_rec.append(np.min([lr, mr, hr, p35r]))
		max_rec.append(np.max([lr, mr, hr, p35r]))

	min_pre, max_pre = [], []
	for lp, mp, hp, p35p in zip(GZX_low_run['precision'], GZX_baseline_run['precision'],
						  		GZX_high_run['precision'], GZX_p35_run['precision']):
		min_pre.append(np.min([lp, mp, hp, p35p]))
		max_pre.append(np.max([lp, mp, hp, p35p]))


	acc = ax.fill_between(days, min_acc, max_acc, color='k', alpha=0.4, lw=3)
	ax.plot(days, GZX_baseline_run['accuracy'], color='k', ls='--', alpha=0.5, lw=3)

	rec = ax.fill_between(days, min_rec, max_rec, color='grey',alpha=0.6, lw=3)
	ax.plot(days, GZX_baseline_run['recall'], color='k', ls='--', alpha=0.5, lw=3)
	#ax.plot(days, GZX_p35_run['recall'], color='darkred', ls='--', alpha=0.5, lw=2)
	#ax.plot(days, GZX_high_run['recall'], color='green', ls='--', alpha=0.5, lw=3)
	#ax.plot(days, GZX_low_run['recall'], color='purple', ls='--', alpha=0.5, lw=3)

	pre = ax.fill_between(days, min_pre, max_pre, color='lightgrey', alpha=0.7, lw=3)
	ax.plot(days, GZX_baseline_run['precision'], color='k', ls='--', alpha=0.5, lw=3)
	#ax.plot(days, GZX_p35_run['precision'], color='darkred', ls='--', alpha=0.5, lw=3)
	#ax.plot(days, GZX_high_run['precision'], color='green', ls='--', alpha=0.5, lw=3)
	#ax.plot(days, GZX_low_run['precision'], color='purple', ls='--', alpha=0.5, lw=3)

	ax.legend((acc, rec, pre), ('Accuracy', 'Completeness', 'Purity'), 
			  loc='lower left')
	ax.set_ylabel('Per cent')
	ax.set_ylim(0.75, 1.0)
	ax.set_xlim(0,num_days-1)

	if single_figure:
		ax.set_xlabel('Days in GZ2 project')
		plt.savefig("GZX_evaluation_%s.png"%outfile)
		plt.show()
		plt.close()
	else:
		# If we're passing axes -- remove tick labeling on x axis
		ax.xaxis.set_ticklabels([])



###############################################################################
#			 CUMULATIVE RETIREMENT : SPREAD DUE TO MULTI SIMS
###############################################################################

def plot_GZX_cumulative_retirement_spread(num_days, low_run, mid_run, high_run,
										  p35_run, gz2_retired, outname, ax=None):
	single_figure = False

	days = np.arange(num_days)

	if not ax:
		fig = plt.figure(figsize=(10,8))
		ax = fig.add_subplot(111)
		single_figure = True

	ax.grid(ls='dotted',lw=0.5)

	low_ret = low_run.computeCumulativeRetiredSubjects(kind='both')
	mid_ret = mid_run.computeCumulativeRetiredSubjects(kind='both')
	high_ret = high_run.computeCumulativeRetiredSubjects(kind='both')
	p35_ret = p35_run.computeCumulativeRetiredSubjects(kind='both')

	min_ret, max_ret = [], []
	for lc, mc, hc, p35c in zip(low_ret, mid_ret, high_ret, p35_ret):
		min_ret.append(np.min([lc, mc, hc]))
		max_ret.append(np.max([lc, mc, hc]))


	# PLot the variation in the cumulative retirement 
	# due to varying initial conditions ---------------------------------
	ax.fill_between(days, min_ret, max_ret, color='steelblue',
					alpha=0.5, lw=3, label='SWAP')
	
	# Plot the baseline cumulative retirement
	ax.plot(days, mid_ret, color='steelblue', lw=3, ls='--')
	#qax.plot(days, p35_ret, color='darkred', lw=2, ls='--')

	# Plot GZ2 cumulative retirement -------------------------------------
	ax.plot(days, gz2_retired, color='darkblue', lw=3, label='Galaxy Zoo 2')


	ax.set_xlim(0,num_days-1)
	ax.set_ylim(0,260000)

	x = min_ret
	scale_y = 1e3
	ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
	ax.yaxis.set_major_formatter(ticks_y)

	ax.set_xlabel('Days in GZ2 Project')
	ax.set_ylabel(r'Cumulative retired subjects  [$10^3$]')

	ax.legend(loc='upper left', frameon=True)

	if single_figure:
		plt.savefig('GZX_cumulative_retirement_{0}'.format(outname))
		plt.show()
		plt.close()



###############################################################################
#			THE MONEY PLOT
###############################################################################

def MONEYPLOT(num_days, mid_run, mid_eval, gz2_retired, combo_eval, MLbureau, outfile):

	"""
	GOAL: plot GZ2, SWAP-only, and GZX retirement rates
	"""
	import swap
	meta = swap.read_pickle('GZ2_sup_PLPD5_p5_flipfeature2b_metadata.pickle', 'metadata')
	sample = meta.subjects

	sample_feat = np.sum(sample['GZ2_raw_combo'] == 0)
	sample_not = np.sum(sample['GZ2_raw_combo'] == 1)

	dates = [datetime(2009, 02, 12)+timedelta(days=i) for i in range(num_days)]
	dates = np.array([datetime.strftime(d, '%Y-%m-%d_%H:%M:%S') for d in dates])
	
	machine = MLbureau.member['RF_accuracy']
	trainhistory = machine.traininghistory
	testhistory = machine.evaluationhistory

	days = np.arange(num_days)
	mid_ret = mid_run.computeCumulativeRetiredSubjects(kind='both')

	# combo_eval is # per day separated into valid, train, & test samples
	# NOT CUMULATIVE -- add it up!
	tps, fps, tns, fns = 0,0,0,0
	for k, v in combo_eval.iteritems():
		combo_eval[k] = np.cumsum(np.array(v, dtype='float64'))
		if 'tps' in k:
			tps+=np.array(combo_eval[k], dtype='float64')
		if 'fps' in k:
			fps+=np.array(combo_eval[k], dtype='float64')
		if 'tns' in k:
			tns+=np.array(combo_eval[k], dtype='float64')
		if 'fns' in k:
			fns+=np.array(combo_eval[k], dtype='float64')

	swap_tps = combo_eval['swap_tps'] + combo_eval['valid_tps']
	swap_fps = combo_eval['swap_fps'] + combo_eval['valid_fps']
	swap_tns = combo_eval['swap_tns'] + combo_eval['valid_tns']
	swap_fns = combo_eval['swap_fns'] + combo_eval['valid_fns']
	swap_total = combo_eval['swap_total'] + combo_eval['valid_total']

	swap_acc = (swap_tps + swap_tns) / swap_total
	swap_rec = (swap_tps) / (swap_tps + swap_fns)
	swap_prec = (swap_tps) / (swap_tps + swap_fps)

	machine_acc = (combo_eval['machine_tps'] + combo_eval['machine_tns']) / combo_eval['machine_total']
	machine_rec = (combo_eval['machine_tps']) / (combo_eval['machine_tps'] + combo_eval['machine_fns'])
	machine_prec = (combo_eval['machine_tps']) / (combo_eval['machine_tps'] + combo_eval['machine_fps'])

	total_accuracy = (tps+tns) / combo_eval['total_retired']
	total_recall = tps / (tps + fns)
	total_precision = tps / (tps + fps)

	short_days = np.arange(len(total_accuracy))


	# ----------------------------------------------------------------
	#  		PLOT INDIVIDUAL COMPONENTS
	# ----------------------------------------------------------------
	"""
	fig = plt.figure(figsize=(9,16))
	gs = gridspec.GridSpec(2,1, wspace=0.1, hspace=0.01)

	# -------- PLOT QUALITY METRICS ----------------------------
	ax = fig.add_subplot(gs[0])
	ax.grid(linestyle='dotted',linewidth=.5)

	acc = ax.plot(short_days, swap_acc, color='red', lw=3, label='SWAP: accuracy')
	rec = ax.plot(short_days, swap_rec, color='red', ls='--', lw=3, label='SWAP: completeness')
	pre = ax.plot(short_days, swap_prec, color='red', ls='-.', lw=3, label='SWAP: purity')

	acc = ax.plot(short_days, machine_acc, color='blue', lw=3, label='RF: accuracy')
	rec = ax.plot(short_days, machine_rec, color='blue', ls='--', lw=3, label='RF: completeness')
	pre = ax.plot(short_days, machine_prec, color='blue', ls='-.', lw=3, label='RF: purity')

	ax.legend(loc='lower left', ncol=2)

	ax.set_ylabel('Proportion')
	ax.set_ylim(0.5, 1.0)
	ax.set_xlim(0, len(short_days)-1)

	# No x tick labels cuz this panel is on top
	ax.xaxis.set_ticklabels([])
	"""

	# -------- PLOT CUMULATIVE RETIREMENT -----------------------
	fig = plt.figure(figsize=(10,15))
	gs = gridspec.GridSpec(3,2, wspace=0.1, hspace=0.01)

	ax = fig.add_subplot(gs[:2, :2])
	ax.grid(linestyle='dotted',linewidth=.5)

	gzx_tot = ax.plot(short_days, combo_eval['total_retired'], color='k')
	swap_only = ax.plot(short_days, mid_ret[:len(short_days)], color='grey', ls='--')
	gzx_rf = ax.plot(short_days, combo_eval['machine_total'], color='b',)
	gzx_swap = ax.plot(short_days, combo_eval['swap_total']+combo_eval['valid_total'], 
					   color='r')

	ax.set_xlabel('Days in GZ2 Project')
	ax.set_ylabel(r'Cumulative retired subjects [$10^3$]')
	ax.set_xlim(0, len(short_days)-1)
	ax.set_ylim(0, 260000)

	x = mid_ret
	scale_y = 1e3
	ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
	ax.yaxis.set_major_formatter(ticks_y)

	ax.legend((gzx_tot[0], gzx_rf[0], gzx_swap[0], swap_only[0]), 
			  ('GZX: Total', 'GZX: RF', 'GZX: SWAP', 'SWAP only'), loc='best')
	


	ax = fig.add_subplot(gs[2, 0])
	ax.grid(linestyle='dotted',linewidth=.5)
	
	plt.plot((swap_tps+combo_eval['machine_tps'])/sample_feat, color='k')
	plt.plot(swap_tps/sample_feat, color='r')
	plt.plot(combo_eval['machine_tps']/sample_feat, color='b')
	plt.ylabel("Fraction")
	plt.xlabel('Days in GZ2 project')
	ax.set_xlim(0, len(short_days)-1)
	ax.set_ylim(0, 0.7)
	ax.text(2, 0.61, 'Featured')

	ax = fig.add_subplot(gs[2, 1])
	ax.grid(linestyle='dotted',linewidth=.5)

	plt.plot((swap_tns+combo_eval['machine_tns'])/sample_not, color='k')
	plt.plot(swap_tns/sample_not, color='r')
	plt.plot(combo_eval['machine_tns']/sample_not, color='b')
	plt.xlabel('Days in GZ2 project')
	ax.yaxis.set_ticklabels([])
	ax.set_xlim(0, len(short_days)-1)
	ax.set_ylim(0, .7)
	ax.text(2, 0.61, 'Not Featured')

	gs.tight_layout(fig)
	plt.savefig('{}_GZX_component_contributions.png'.format(outfile))
	plt.show()

	pdb.set_trace()



	# ----------------------------------------------------------------
	#  		PLOT COMBINED GZX RESULTS
	# ----------------------------------------------------------------
	fig = plt.figure(figsize=(11,16))
	gs = gridspec.GridSpec(2,1, wspace=0.1, hspace=0.01)


	#  		PLOT QUALITY METRICS 
	# ----------------------------------------------------------------
	ax = fig.add_subplot(gs[0])
	ax.grid(linestyle='dotted',linewidth=.5)


	# plot the SWAP-ONLY quality metrics
	ax.plot(days, mid_eval['accuracy'], color='steelblue', lw=3)
	ax.plot(days, mid_eval['recall'], color='steelblue', ls='--', lw=3)
	ax.plot(days, mid_eval['precision'], color='steelblue', ls='-.', lw=3)

	# plot FULL GZX metrics
	short_days = np.arange(len(total_accuracy))
	acc = ax.plot(short_days, total_accuracy, color='darkred', lw=3)
	rec = ax.plot(short_days, total_recall, color='darkred', ls='--', lw=3)
	pre = ax.plot(short_days, total_precision, color='darkred', ls='-.', lw=3)

	ax.legend((acc[0], rec[0], pre[0]), ('Accuracy', 'Completeness', 'Purity'), 
			  loc='lower right')
	ax.set_ylabel('Proportion')
	ax.set_ylim(0.5, 1.0)
	ax.set_xlim(0, len(days)-1)

	# No x tick labels cuz this panel is on top
	ax.xaxis.set_ticklabels([])


	#  		PLOT REITREMENT RATES
	# ----------------------------------------------------------------
	ax = fig.add_subplot(gs[1])
	ax.grid(linestyle='dotted',linewidth=.5)


	# Plot the original SWAP-onlyl run
	swap = ax.plot(days, mid_ret, color='steelblue', lw=3, ls='-')
	
	# Plot GZ2 
	gz2 = ax.plot(days, gz2_retired, color='darkblue', lw=3)

	# Plot the FULL GZX run
	combo_cum_ret = combo_eval['total_retired']
	new_days = np.arange(len(combo_cum_ret))
	gzx = ax.plot(new_days, combo_cum_ret, color='darkred',lw=3, ls='-')
	
	first_day_training = np.where(trainhistory['At_Time'][0] == dates)[0]
	ax.axvline(x=first_day_training, ls='--', lw=1, color='k')
	ax.text(8,240000, 'Machine starts training',fontsize='20')


	ax.set_xlim(0, len(days)-1)
	ax.set_ylim(0,260000)

	x = mid_ret
	scale_y = 1e3
	ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
	ax.yaxis.set_major_formatter(ticks_y)
	
	ax.legend((gzx[0], swap[0], gz2[0]), ('GZ Express','SWAP only', 'Galaxy Zoo 2'), 
			  loc='center right')

	ax.set_xlabel('Days in GZ2 Project')
	ax.set_ylabel(r'Cumulative retired subjects [$10^3$]')

	plt.savefig('{}_moneyplot.png'.format(outfile))

	plt.show()

	""" used this shitty way for galmorph talk
	F = open('best_sample_size.pickle','rb')
	best_sample_size = cPickle.load(F)
	best_cum = np.cumsum(best_sample_size)

	MLbureau = swap.read_pickle('GZ2_sup_PLPD5_p5_flipfeature2b_MLbureau.pickle','bureau')
	machine=MLbureau.member['RF_accuracy']
	pdb.set_trace()
	new_path = mid_ret[12:12+len(best_cum)]
	nn = new_path[1:]+best_cum
	new_path[1:] = nn
	new_days = days[12:12+len(best_cum)]
	"""

###############################################################################
#			SCATTER PLOT : 		USER PROBABILITIES
###############################################################################

def plot_user_probabilities(bureau, number_of_users):

	number_of_users = 1000

	# Gather bureau/agent data
	# ----------------------------------------------------------------------
	bureau_info = collect_probabilities(bureau)

	TheseFewNames = bureau.shortlist(np.min([number_of_users, bureau.size()]))
	index = [i for i,Name in enumerate(bureau.list()) if Name in set(TheseFewNames)]
		
	PD = bureau_info['PDarray'][index]
	PL = bureau_info['PLarray'][index]
	Ntraining = bureau_info['Ntraining'][index]

	PD_full = bureau_info['PDarray']
	PL_full = bureau_info['PLarray']

	# Prepare the figure
	# ----------------------------------------------------------------------
	bins = np.linspace(0.0, 1.0, 20, endpoint=True) # Bins for HISTOGRAMS
	pmin, pmax = 0., 1.

	fig = plt.figure(figsize=(10,10))
	gs = gridspec.GridSpec(4,4, wspace=0., hspace=0.)

	scatterax = fig.add_subplot(gs[1:,:3])
	lowerhistax = fig.add_subplot(gs[0,:3])
	righthistax = fig.add_subplot(gs[1:,-1])

	# --------------------------  SCATTER PLOT  -------------------------------
	# Axis details
	scatterax.set_xlim(pmin, pmax)
	scatterax.set_ylim(pmin, pmax)
	scatterax.set_ylabel('Pr("NOT"|NOT)')
	scatterax.set_xlabel('Pr("FEAT"|FEAT)')

	# User Type names and line boundaries
	scatterax.axvline(0.5, color='grey', linestyle='dotted', alpha=0.5)
	scatterax.axhline(0.5, color='grey', linestyle='dotted', alpha=0.5)
	scatterax.plot([0,1], [1,0], color='grey', alpha=0.5)

	scatterax.text(0.02,0.02,'"Obtuse"',color='dimgray')
	scatterax.text(0.02,0.95,'"Pessimistic"',color='dimgray')
	scatterax.text(0.73,0.02,'"Optimistic"',color='dimgray')
	scatterax.text(0.80,0.95,'"Astute"',color='dimgray')
	scatterax.text(0.15,0.87,'"Random classifier"',color='dimgray',rotation=-45)

	# Training received:
	size = 4*Ntraining + 6.0
	scatterax.scatter(PL, PD, s=size, color='purple', alpha=0.5)

	# --------------------------  RIGHT HIST  -------------------------------
	# Axis details
	#righthistax.set_xlim(0.5*len(PD),0.0)
	righthistax.set_ylim(pmin,pmax)
	righthistax.set_xticklabels([])
	righthistax.set_yticklabels([])

	righthistax.axhline(0.5, color='gray', linestyle='dotted')
	righthistax.hist(PD_full, bins=bins, orientation='horizontal', color='tomato', 
					 histtype='stepfilled', alpha=0.7)

	# --------------------------  UPPER HIST  -------------------------------
	# Axis details
	#lowerhistax.set_xlim(pmin,pmax)
	#lowerhistax.set_ylim(0.5*len(PD),0.0)
	lowerhistax.set_xticklabels([])
	lowerhistax.set_yticklabels([])

	lowerhistax.axvline(0.5, color='gray', linestyle='dotted')
	lowerhistax.hist(PL_full, bins=bins, histtype='stepfilled', color='yellow', alpha=0.7)

	plt.savefig('test_user_probs.png',dpi=300)
	plt.show()



###############################################################################
#			PLOT   VOTE   DISTRIBUTIONS
###############################################################################
def plot_vote_distributions(gz2_baseline, mid_sim):

	gz2_clicks = gz2_baseline['total_classifications']

	#"""
	fig = plt.figure(figsize=(8,6))
	ax = fig.add_subplot(111)
	ax.set_xlabel('Classifications till retirement')
	ax.set_ylabel('Proportion')

	mid_sim.plot_clicks_till_retired(ax, plot_combined=True)

	weights = np.ones_like(gz2_clicks, dtype='float64')/len(gz2_clicks)
	ax.hist(gz2_clicks, weights=weights, 
			bins=np.arange(0, np.max(gz2_clicks)+1, 1), 
			color='darkblue', histtype='stepfilled', alpha=0.5,
			label='GZ2')
	ax.set_ylim(0,0.15)

	ax.legend(loc='upper center', frameon=False)
	
	"""
	# ---------------- Plot votes against each other ----------------------

	xbins = np.arange(0, np.max(gz2_clicks)+1, 1)
	ybins = np.arange(0, np.max(all_retired['Nclass'])+1, 1)

	xweights = np.ones_like(gz2_clicks) / len(gz2_clicks)
	yweights = np.ones_like(all_retired['Nclass']) / len(all_retired)

	H, xedges, yedges = np.histogram2d(gz2_clicks, all_retired['Nclass'], 
									   bins=(xbins, ybins))

	#fig = plt.figure(figsize=(8,6))
	ax = fig.add_subplot(122)
	ax.set_title('Classifications till retirement')

	X, Y = np.mgrid[xedges[0]:xedges[-1]:1, yedges[0]:yedges[-1]:1]
	#im = ax.imshow(H.T, interpolation='nearest', origin='low',
	#			   cmap='Blues')
	ax.contour(X.T, Y.T, H.T, lw=2, cmap='Blues')

	ax.set_xlabel('GZ2')
	ax.set_ylabel('SWAP')

	ax.set_xlim(25, 60)
	ax.set_ylim(0, 25)

	"""
	plt.savefig('GZX_clicks_till_retired_baseline.png')
	plt.show()

	pdb.set_trace()




###############################################################################
#			PLOT  WHERE  SWAP  AND  GZ2  DISAGREE (FNS + FPS)
###############################################################################
def swap_gets_it_wrong(fps, fns, full_SWAP):

	"""
	INPUTS:
		fps 	False Positives (classified as F but actually S)
		fns 	False Negatives (classified as S but actually F)
		full_SWAP 	catalog of all retired subjects

		Note that all catalogs must have previously been joined with 
		gz2 metadata and/or gz2 classification catalog
	"""

	fps_color = 'blue'
	fns_color = 'red'

	fig = plt.figure(figsize=(15,8))
	gs = gridspec.GridSpec(1,2)


	# Is there a difference in the SIZE/magnitude distribution?
	ax = fig.add_subplot(gs[1])

	#"""
	xbins = np.arange(10, 18., 0.5)
	ybins = np.arange(0, 26, 1)

	H, x_, y_ = np.histogram2d(full_SWAP['PETROMAG_R']-full_SWAP['EXTINCTION_R'],
							   full_SWAP['PETROR50_R'], bins=(xbins, ybins))

	X, Y = np.mgrid[xbins[0]:xbins[-1]:0.5, ybins[0]:ybins[-1]:1]
	
	ax.contour(X, Y, H, colors='k', lw=2, levels=[1e4, 1e3, 1e2, 1e1])


	Hfps, x_, y_ = np.histogram2d(fps['PETROMAG_R']-fps['EXTINCTION_R'], 
								  fps['PETROR50_R'], bins=(xbins, ybins))


   	ax.contour(X, Y, Hfps, colors=fps_color, lw=2, levels=[1e4, 1e3, 1e2, 1e1])

	ax.scatter(fns['PETROMAG_R']-fns['EXTINCTION_R'], fns['PETROR50_R'], 
    		   color=fns_color, alpha=.6, marker='.')

	ax.axhline(5., ls='--', color='k', lw=0.5)

	small_tot = np.sum(full_SWAP['PETROR50_R'] <= 5.)
	small_fps = np.sum(fps['PETROR50_R'] <= 5.)
	small_fns = np.sum(fns['PETROR50_R'] <= 5.)

	#fraction = 100*float(small_fps + small_fns) / (len(fps)+len(fns))
	#ax.text(15., 20., "%.2f%%"%fraction)

	ax.set_ylabel(r'$R_{50}$ (arcsec)')
	ax.set_xlabel('r magnitude')
	ax.set_ylim(0., 20.)
	ax.set_xlim(10., 17.5)
	#"""

	# Distribution of Features_or_Disk fraction? 
	ax = fig.add_subplot(gs[0])

	disk_fraction = 't01_smooth_or_features_a02_features_or_disk_fraction'
	star_fraction = 't01_smooth_or_features_a03_star_or_artifact_fraction'
	smooth_frac = 't01_smooth_or_features_a01_smooth_fraction'

	full_SWAP['combo'] = full_SWAP[disk_fraction]+full_SWAP[star_fraction]
	sub_sample = full_SWAP[~np.isnan(full_SWAP['combo'])]

	fns['combo'] = fns[disk_fraction] + fns[star_fraction]
	fps['combo'] = fps[disk_fraction] + fps[star_fraction]


	#sub_full_SWAP = full_SWAP[disk_fraction][~np.isnan(full_SWAP[disk_fraction])]
	bins = np.arange(0., 1.+.05, .05)

	weights = np.ones_like(sub_sample['combo'],dtype='float64')/len(sub_sample)
	ax.hist(sub_sample['combo'], bins=bins, weights=weights, color='white', 
			histtype='stepfilled', label='SWAP', lw=2)

	weights = np.ones_like(fns['combo'],dtype='float64')/len(fns)
	ax.hist(fns['combo'][~np.isnan(fns['combo'])], bins=bins, weights=weights, 
    		color=fns_color, alpha=.6, histtype='stepfilled', label='False Negatives')

	weights = np.ones_like(fps['combo'],dtype='float64')/len(fps)
	ax.hist(fps['combo'][~np.isnan(fps['combo'])], bins=bins, weights=weights, 
    		color=fps_color, alpha=.6, histtype='stepfilled', label='False Positives')

	ax.legend(loc='upper left', frameon=False)

	ax.set_xlabel(r'GZ2 $f_{features} + f_{artifact}$')
	ax.set_ylabel('Proportion')
	ax.set_xlim(0, 1.)
	ax.set_ylim(0.,0.7)

	plt.tight_layout()
	plt.savefig('swapgetsitwrong_test.png')
	plt.show()

	pdb.set_trace()


###############################################################################
#			PLOT  DISTRIBUTIONS OF MORPHOLOGY PARAMS 
###############################################################################
def plot_morph_params_1D(machine_retired, machine_not_retired, metadata, outfile, **kwargs):
	"""
	1D distributions of the morphologly parameters: G, M20, C, A, elipticity
	"""

	# FIRST : plot FEAT vs NOT of all MACHINE-RETIRED SUBJECTS
	# SECOND : plot FALSE POS vs FALSE NEG of the MACHINE-RETIRED SUBJECTS
	# THIRD : plot RETIRED vs NOT RETIRED of the MACHINE-RETIRED SUBJECTS


	machine_feat = machine_retired[machine_retired['machine_probability']>=0.9]
	machine_not = machine_retired[machine_retired['machine_probability']<=0.1]

	machine_fps = machine_feat[machine_feat['GZ2_raw_combo']==1]
	machine_fns = machine_not[machine_not['GZ2_raw_combo']==0]


	subsets2 = [machine_feat, machine_fps, machine_not_retired]
	subsets1 = [machine_not, machine_fns, machine_retired]

	colorset2 = ['yellow', 'blue', 'purple']
	colorset1 = ['tomato', 'red', 'steelblue']

	labels2 = ["'Featured'", 'False Positives', 'Not Retired']
	labels1 = ["'Not'", 'False Negatives', 'Retired']

	fig = plt.figure(figsize=(18,12))
	gs = gridspec.GridSpec(3,5, wspace=0.05, hspace=0.05)

	i = 0
	for subset1, subset2 in zip(subsets1, subsets2):

		ax = fig.add_subplot(gs[0+i,0])
		ax.hist(metadata['G'], bins=30, normed=True, range=(.4,.8),alpha=0.5,
				histtype='step', color='black', lw=3)
		ax.hist(subset1['G'], bins=30, normed=True, range=(.4,.8),alpha=0.5, 
				histtype='stepfilled', color=colorset1[i])
		ax.hist(subset2['G'], bins=30, normed=True, range=(.4,.8),alpha=0.5,
				histtype='stepfilled', color=colorset2[i])

		if i!=2:
			ax.set_xticklabels([])
		else:
			ax.set_xlabel(r'$G$')
			ax.set_xlim(0.4, 0.8)
			ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
		ax.set_yticklabels([])
		ax.set_ylabel('Normalized Distribution')

		ax = fig.add_subplot(gs[0+i,1])
		ax.hist(metadata['M20'], bins=30, normed=True, range=(-3., -1.), alpha=0.5,
				histtype='step', color='black', lw=3)
		ax.hist(subset1['M20'], bins=30, normed=True, range=(-3., -1.), alpha=0.5, 
				histtype='stepfilled', color=colorset1[i])
		ax.hist(subset2['M20'], bins=30, normed=True, range=(-3., -1.), alpha=0.5,  
				histtype='stepfilled', color=colorset2[i])

		if i!=2:
			ax.set_xticklabels([])
		else:
			ax.set_xlabel(r'M$_{20}$')
			ax.set_xlim(-1., -3.0)
			ax.set_xticks([-1.5, -2.0, -2.5])
		ax.set_yticklabels([])

		ax = fig.add_subplot(gs[0+i,2])
		ax.hist(metadata['C'], bins=30, normed=True, range=(1.5, 5.5), alpha=0.5,
				histtype='step', color='black', lw=3)
		ax.hist(subset1['C'], bins=30, normed=True, range=(1.5, 5.5), alpha=0.5, 
				histtype='stepfilled', color=colorset1[i])
		ax.hist(subset2['C'], bins=30, normed=True, range=(1.5, 5.5), alpha=0.5, 
				histtype='stepfilled', color=colorset2[i])

		if i!=2:
			ax.set_xticklabels([])
		else:
			ax.set_xlabel(r'$C$')
			ax.set_xlim(1.5, 5.5)
			ax.set_xticks([2, 3, 4, 5])
		ax.set_ylim(0, 1.4)
		ax.set_yticklabels([])


		ax = fig.add_subplot(gs[0+i,3])
		ax.hist(metadata['E'], bins=30,normed=True, range=(0,1), alpha=0.5,
				histtype='step', color='black', lw=3)
		ax.hist(subset1['E'], bins=30,normed=True, range=(0,1), alpha=0.5, 
				histtype='stepfilled', color=colorset1[i])
		ax.hist(subset2['E'], bins=30,normed=True, range=(0,1), alpha=0.5, 
				histtype='stepfilled', color=colorset2[i])

		if i!=2:
			ax.set_xticklabels([])
		else:
			ax.set_xlabel(r'$1-b/a$')
			ax.set_xlim(0, 1.)
			ax.set_xticks([.2, .4, .6, .8])
		ax.set_yticklabels([])


		ax = fig.add_subplot(gs[0+i,4])
		ax.hist(metadata['A'], bins=30, normed=True, range=(0.,0.4),alpha=0.5,
				histtype='step', color='black', lw=3, label='GZ2 catalog')
		ax.hist(subset1['A'], bins=30, normed=True, range=(0.,0.4),alpha=0.5, 
				histtype='stepfilled', color=colorset1[i], label=labels1[i])
		ax.hist(subset2['A'], bins=30, normed=True, range=(0.,0.4),alpha=0.5, 
				histtype='stepfilled', color=colorset2[i], label=labels2[i])

		if i==0:
			ax.legend(loc='upper right', fontsize=16)
		else:
			handles, labels = ax.get_legend_handles_labels()
			ax.legend(handles[1:], labels[1:], loc='upper right', fontsize=16)

		if i==1:
			ax.text(.14, 13., '(GZ2 raw labels)', fontsize=14)

		if i!=2:
			ax.set_xticklabels([])
		else:
			ax.set_xlabel(r'$A$')
			ax.set_xlim(0., .4)
			ax.set_xticks([0.05, 0.15, 0.25, 0.35])
		ax.set_yticklabels([])

		i += 1

	#gs.tight_layout(fig)
	plt.savefig('{}_morph_params_raw_labels_4paper.png'.format(outfile))
	plt.show()

	pdb.set_trace()


###############################################################################
#			MAIN
###############################################################################
def main():
	"""
	This script makes ALL THE MUTHAFUCKIN FIGURES FOR MAH PAYPAH.

	1. VOLUNTEER PROBABILITIES
		plot_user_probabilities()
		REQUIRES: swap bureau file and # of users to plot

	2. VOTE DISTRIBUTIONS COMPARED TO GZ2
		plot_vote_distributions()
		REQUIRES: gz2_metadata and simulation to compare to
	
	3. BASELINE SWAP SIMULATION COMPARED TO GZ2
		plot_GZX_baseline()
		REQUIRES: 	baseline simulation, evaluation ascii file for baseline run,
				  	gz2_retired (cumulative retired subjects in GZ2)
		NOTES: 		this plots the retired subject rate AND the corresponding 
					quality metrics ON THE SAME AXES

					The eval file and the GZ2 retired subjects file must be 
					created	in separate scripts! (update later with which ones)
	
	4. VARIATIONS IN SWAP
		plot_GZX_evaluation_spread()
		plot_GZX_cumulative_retirement_spread()

		REQUIRES: 	three simulations to compare (for spread in retirement)
					three evaluation files to compare (for spread in eval)
		NOTES:		these 

	5. SWAP AND GZ2 DISAGREE 
		swap_gets_it_wrong()

	6. MONEYPLOT
		MONEYPLOT()
	"""


	make_volunteer_probabilties_plot = False 
	make_vote_distributions_plot = False 
	make_baseline_simulation_plot = False 
	make_swap_variations_plot = False  
	make_swap_gets_it_wrong_plot = False
	make_moneyplot = False
	make_morph_distributions_plot = True


	# Load up some GZ2 data
	# -----------------------------------------------
	gz2_metadata = Table.read('metadata_ground_truth_labels.fits')
	if 'GZ2_deb_combo' not in gz2_metadata.colnames:
		gz2_metadata['GZ2_raw_combo'] = GZ2_label_SMOOTH_NOT(bigfuckingtable,type='raw')
		gz2_metadata.write('metadata_ground_truth_labels.fits', overwrite=True)


	gz2_metadata['zooid'] = gz2_metadata['SDSS_id']
	gz2_metadata['id'] = gz2_metadata['asset_id']

	F = open('GZ2_cumulative_retired_subjects_expert.pickle','r')
	gz2_cum_sub_retired = cPickle.load(F)


	# Load up BASELINE simulation
	# ------------------------------------------------------
	mid_name = 'sup_PLPD5_p5_flipfeature2b_norandom2'
	#stuff = generate_SWAP_eval_report(mid_sim, gz2_metadata, outname=mid_name+'_raw_combo',
	#								   write_file=True, gz_kind='raw_combo')
	mid_eval2 = Table.read('GZX_evaluation_{0}.txt'.format(mid_name+'_raw_combo'), format='ascii')

	mid_sim = Simulation(config='update_sup_PLPD5_p5_flipfeature2b_norandom2.config',
						 directory='.',
						 variety='feat_or_not')


	""" MAKE VOLUNTEER PROBABILTIES PLOT """
	if make_volunteer_probabilties_plot:
		import swap
		bureau = swap.read_pickle('GZ2_sup_PLPD5_p5_flipfeature2b_bureau.pickle', 'bureau')
		plot_user_probabilities(bureau, 200)


	""" MAKE BASELINE SIMULATION PLOT """
	if make_baseline_simulation_plot:
		plot_GZX_baseline(mid_sim, mid_eval2, gz2_cum_sub_retired)


	""" MAKE MONEY PLOT """
	if make_moneyplot:

		outfile = 'GZ2_sup_PLPD5_p5_flipfeature2b_RF_accuracy_redo_raw_combo'
		
		# this file made by explore_MLagents.py
		F = open('{}_combo_analysis.pickle'.format(outfile))
		combo_run = cPickle.load(F)
		F.close()

		# Load up the Machine bureau
		F = open('GZ2_sup_PLPD5_p5_flipfeature2b_MLbureau.pickle')
		MLbureau = cPickle.load(F)
		F.close()

		MONEYPLOT(92, mid_sim, mid_eval2, gz2_cum_sub_retired, combo_run, MLbureau, outfile=outfile)


	if make_morph_distributions_plot:

		outfile = 'GZ2_sup_PLPD5_p5_flipfeature2b_RF_accuracy_redo_raw_combo'
		machine_retired = Table.read('{}_machine_retired_subjects.fits'.format(outfile))
		machine_not_retired = Table.read('{}_machine_not_retired_subjects.fits'.format(outfile))


		

		plot_morph_params_1D(machine_retired, machine_not_retired, gz2_metadata, outfile)


	""" MAKE SWAP GETS IT WRONG PLOT """
	if make_swap_gets_it_wrong_plot:
		bigfuckingtable = Table.read('../GZ2ASSETS_NAIR_MORPH_MAIN.fits')
		gz2_bigfuckingtable = join(gz2_metadata, bigfuckingtable, keys='id')
		
		all_retired = mid_sim.fetchCatalog(mid_sim.retiredFileList[-1])
		gz2_baseline = join(gz2_bigfuckingtable, all_retired, keys='zooid')

		tps2, fps2, tns2, fns2 = calculate_confusion_matrix(gz2_baseline[gz2_baseline['P']>0.3],
											   		gz2_baseline[gz2_baseline['P']<0.3],
							   						smooth_or_not=False, gz_kind='raw')
					   					
		swap_gets_it_wrong(fps, fns, gz2_baseline)


	""" MAKE VOTE DISTRIBUTION PLOT """
	if make_vote_distributions_plot:
		plot_vote_distributions(gz2_metadata, mid_sim)


	""" MAKE SWAP VARIATIONS PLOT(S) """
	if make_swap_variations_plot:

		# Load up simulations varying subject PRIOR
		# -------------------------------------------------------
		low_p = 'sup_PLPD5_p2_flipfeature2_norand'
		high_p = 'sup_PLPD5_p8_flipfeature2_norand'
		p35 = 'sup_PLPD5_p35_flipfeature2_norand'

		low_p_eval2 = Table.read('GZX_evaluation_{0}.txt'.format(low_p+'_raw_combo'), format='ascii')
		high_p_eval2 = Table.read('GZX_evaluation_{0}.txt'.format(high_p+'_raw_combo'), format='ascii')
		p35_eval2 = Table.read('GZX_evaluation_{0}.txt'.format(p35+'_raw_combo'), format='ascii')

		low_p_sim = Simulation(config='update_sup_PLPD5_p2_flipfeature2_norand.config',
							   directory='S_PLPD5_p2_ff_norand/',
							   variety='feat_or_not')

		high_p_sim = Simulation(config='update_sup_PLPD5_p8_flipfeature2_norand.config',
								directory='S_PLPD5_p8_ff_norand/',
								variety='feat_or_not')

		p35_sim = Simulation(config='update_sup_PLPD5_p35_flipfeature2_norand.config',
							directory='S_PLPD5_p35_ff_norand/',
							variety='feat_or_not')


		# Load up simulations for varying user PL/PD
		# -------------------------------------------------------
		low_plpd = 'sup_PLPD4_p5_flipfeature2_norand'
		high_plpd = 'sup_PLPD6_p5_flipfeature2_norand'

		low_plpd_eval2 = Table.read('GZX_evaluation_{0}.txt'.format(low_plpd+'_raw_combo'), format='ascii')
		high_plpd_eval2 = Table.read('GZX_evaluation_{0}.txt'.format(high_plpd+'_raw_combo'), format='ascii')

		low_plpd_sim = Simulation(config='update_sup_PLPD4_p5_flipfeature2_norand.config',
								  directory='S_PLPD4_p5_ff_norand/',
								  variety='feat_or_not')

		high_plpd_sim = Simulation(config='update_sup_PLPD6_p5_flipfeature2_norand.config',
								   directory='S_PLPD6_p5_ff_norand/',
								   variety='feat_or_not')

		# VARY PRIOR
		fig = plt.figure(figsize=(11,16))
		gs = gridspec.GridSpec(2,1)
		gs.update(wspace=0.1, hspace=0.01)

		ax = fig.add_subplot(gs[0])
		plot_GZX_evaluation_spread(92, low_p_eval2, mid_eval2, high_p_eval2, p35_eval2,
							 	   'compare_PLPD_4paper', ax)

		ax2 = fig.add_subplot(gs[1])
		plot_GZX_cumulative_retirement_spread(92, low_p_sim, mid_sim, high_p_sim, p35_sim,
											  gz2_cum_sub_retired, 'compare_prior_4paper', ax2)

		gs.tight_layout(fig)
		plt.savefig('GZX_eval_and_retirement_prior_spread_4paper_v2.png')
		plt.show()
		plt.close()

		# -----------------------------------------------------------
		# VARY PLPD
		fig = plt.figure(figsize=(11,16))
		gs = gridspec.GridSpec(2,1)
		gs.update(wspace=0.1, hspace=0.01)


		ax = fig.add_subplot(gs[0])
		plot_GZX_evaluation_spread(92, low_plpd_eval2, mid_eval2, high_plpd_eval2, 
							 	   'compare_PLPD_4paper', ax)

		ax2 = fig.add_subplot(gs[1])
		plot_GZX_cumulative_retirement_spread(92, low_plpd_sim, mid_sim, high_plpd_sim, 
											  gz2_cum_sub_retired, 'compare_prior_4paper', ax2)

		gs.tight_layout(fig)
		plt.savefig('GZX_eval_and_retirement_PLPD_spread_4paper_v2.png')
		plt.show()
		plt.close()
		#"""


	# These were created in order to compare the ORDER in which subjects
	# were classified in different SWAP runs
	#baseline_retired = mid_sim.fetchRetiredSubjectsByDate()
	#low_plpd_retired = low_plpd_sim.fetchRetiredSubjectsByDate()
	#high_plpd_retired = high_plpd_sim.fetchRetiredSubjectsByDate()
	


if __name__ == '__main__':
	main()