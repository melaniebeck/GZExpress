 #!/usr/bin/env python -W ignore::DeprecationWarning


from __future__ import division

from simulation import Simulation
from astropy.table import Table, join, vstack
from argparse import ArgumentParser
import numpy as np
import pdb, sys
from datetime import *
import cPickle
import swap
from GZX_SWAP_evaluation import generate_SWAP_eval_report, \
								calculate_confusion_matrix, \
								GZ2_label_SMOOTH_NOT
from GZX_paper_figure_functions import *

###############################################################################
#			MAIN
###############################################################################
def main():
	"""
	This script makes ALL THE MUTHAFUCKIN FIGURES FOR MAH PAYPAH.

	1. VOLUNTEER PROBABILITIES
		NAME
			plot_user_probabilities()

		REQUIRES
			swap bureau file and # of users to plot

	2. VOTE DISTRIBUTIONS COMPARED TO GZ2
		NAME
			plot_vote_distributions()

		REQUIRES
			gz2_metadata and simulation to compare to
	
	3. BASELINE SWAP SIMULATION COMPARED TO GZ2
		NAME
			plot_GZX_baseline()

		REQUIRES 	
			baseline simulation, evaluation ascii file for baseline run,
			gz2_retired (cumulative retired subjects in GZ2)

		NOTES 		
			this plots the retired subject rate AND the corresponding 
			quality metrics ON THE SAME AXES

			The eval file and the GZ2 retired subjects file must be 
			created	in separate script: (generate_SWAP_eval_report)
	
	4. VARIATIONS IN SWAP
		NAME
			plot_GZX_evaluation_spread()
			plot_GZX_cumulative_retirement_spread()

		REQUIRES 	
			three simulations to compare (for spread in retirement)
			three evaluation files to compare (for spread in eval)

		NOTES		
			the eval files have to be created with generate_SWAP_eval_report

	5. SWAP AND GZ2 DISAGREE 
		swap_gets_it_wrong()

	6. MONEYPLOT
		MONEYPLOT()

	7. 1D MORPHOLOGY DISTRIBUTIONS
		NAME
			plot_morph_params_1D()

		REQUIRES

	"""


	make_volunteer_probabilties_plot = False
	make_subject_trajectory_plot = True 
	make_vote_distributions_plot = False 
	make_baseline_simulation_plot = False 
	make_swap_variations_plot = False  
	make_swap_gets_it_wrong_plot = False
	make_moneyplot = False
	make_morph_distributions_plot = False
	make_roc_curves = False
	calculate_GX_human_effort = False


	survey = 'GZ2_sup_PLPD5_p5_flipfeature2b'
	dir_tertiary = 'tertiary_simulation_output'
	dir_sim_machine = 'sims_Machine/redo_first_run_raw_combo/'
	dir_sim_swap = 'sims_SWAP/S_PLPD5_p5_ff_norand/'

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
	mid_eval2 = Table.read('{0}/GZX_evaluation_{1}.txt'.format(dir_tertiary, 
															   mid_name+'_raw_combo'), 
						   format='ascii')

	mid_sim = Simulation(config='configfiles/update_sup_PLPD5_p5_flipfeature2b_norandom2.config',
						 directory=dir_sim_swap,
						 variety='feat_or_not')


	""" MAKE VOLUNTEER PROBABILTIES PLOT """
	if make_volunteer_probabilties_plot:

		# Load up the SWAP Simulation AGENT BUREAU
		picklename = '{0}/{1}_bureau.pickle'.format(dir_sim_swap,survey)
		bureau = swap.read_pickle(picklename, 'bureau')
		plot_user_probabilities(bureau, 200)


	if make_subject_trajectory_plot:

		# Load up the SWAP Simulation AGENT BUREAU
		picklename = '{0}/{1}_collection.pickle'.format(dir_sim_swap,survey)
		collection = swap.read_pickle(picklename, 'collection')
		plot_subject_trajectories(collection, 200)


	""" MAKE BASELINE SIMULATION PLOT """
	if make_baseline_simulation_plot:

		# BASELINE fig requires BASELINE Simulation, 
		#						evaluation output for that sim,
		#						cumulative retirement for GZ2
		plot_GZX_baseline(mid_sim, mid_eval2, gz2_cum_sub_retired)


	""" MAKE MONEY PLOT """
	if make_moneyplot:
		

		outfile = '{}/{}_RF_accuracy_redo_raw_combo'.format(dir_tertiary,survey)
		
		# this file made by explore_MLagents.py
		F = open('{}_combo_analysis.pickle'.format(outfile), 'rb')
		combo_run = cPickle.load(F)
		F.close()

		# Load up the Machine bureau
		F = open('{0}/{1}_MLbureau.pickle'.format(dir_sim_machine, survey),'rb')
		MLbureau = cPickle.load(F)
		F.close()

		MONEYPLOT(92, mid_sim, mid_eval2, gz2_cum_sub_retired, combo_run, MLbureau, outfile=outfile)


	""" MORPH DISTRIBUTIONS """
	if make_morph_distributions_plot:

		# Plotting FEAT vs NOT, FALSE POS & FALSE NEGs, RETIRED vs NOT RETIRED
		# to do all that, need files that were created.... GZX_SWAP_eval?
		outfile = 'GZ2_sup_PLPD5_p5_flipfeature2b_RF_accuracy_redo_raw_combo'
		machine_retired = Table.read('tertiary_simulation_output/{}_machine_retired_subjects.fits'.format(outfile))
		machine_not_retired = Table.read('tertiary_simulation_output/{}_machine_not_retired_subjects.fits'.format(outfile))

		plot_morph_params_1D(machine_retired, machine_not_retired, gz2_metadata, outfile)


	""" MAKE SWAP GETS IT WRONG PLOT """
	if make_swap_gets_it_wrong_plot:

		# Compare SWAP-retired subjects to various parameters in the GZ2 Main Catalog
		bigfuckingtable = Table.read('../SpaceWarps/analysis/GZ2ASSETS_NAIR_MORPH_MAIN.fits')
		gz2_bigfuckingtable = join(gz2_metadata, bigfuckingtable, keys='id')
		
		all_retired = mid_sim.fetchCatalog(mid_sim.retiredFileList[-1])
		gz2_baseline = join(gz2_bigfuckingtable, all_retired, keys='zooid')

		tps2, fps2, tns2, fns2 = calculate_confusion_matrix(gz2_baseline[gz2_baseline['P']>0.3],
											   				gz2_baseline[gz2_baseline['P']<0.3],
							   								smooth_or_not=False, gz_kind='raw_combo')
		
		correct = vstack([tps2, tns2])
		#print len(correct)

		swap_gets_it_wrong(fps2, fns2, correct)



	""" MAKE VOTE DISTRIBUTION PLOT """
	if make_vote_distributions_plot:

		# Requires the Vote Distributions for GZ2 and those from the Simulation
		plot_vote_distributions(gz2_metadata, mid_sim)


	if calculate_GX_human_effort: 

		mlbureaufile = 'sims_Machine/redo_first_run_raw_combo/GZ2_sup_PLPD5_p5_flipfeature2b_MLbureau.pickle'
		MLbureau = swap.read_pickle(mlbureaufile,'bureau')

		machine_meta = 'sims_Machine/redo_first_run_raw_combo/GZ2_sup_PLPD5_p5_flipfeature2b_metadata.pickle'
		all_subjects = swap.read_pickle(machine_meta, 'metadata').subjects

		#subjects = all_subjects[all_subjects['retired_date']!='2016-09-10']
		mclass = all_subjects[all_subjects['MLsample']=='mclas']

		swaps = all_subjects[(all_subjects['MLsample']=='train') | 
							 (all_subjects['MLsample']=='valid')]


		catalog = mid_sim.fetchCatalog(mid_sim.retiredFileList[-1])
		catalog['SDSS_id'] = catalog['zooid']

		# How many machine-retired subjects would have been retired by SWAP anyway? 
		#swap_mach_retired = join(catalog, mclass, keys='SDSS_id')
		swap_retired = join(catalog, swaps, keys='SDSS_id')

		# Assume that only Human Effort came from training sample
		effort = np.sum(swap_retired['Nclass'])
		print effort

		# LOOK AT MOST IMPORTANT FEATURES FOR MACHINE
		machine = MLbureau.member['RF_accuracy']
		trainhist = machine.traininghistory

		models = trainhist['Model']

		for i, model in enumerate(models):

			if i==0:
				feature_importances = model.feature_importances_

			else: 
				feature_importances = np.vstack([feature_importances, 
												 model.feature_importances_])


		labels = ['M$_{20}$', '$C$', '$1-b/a$', '$A$', '$G$']
		fi = feature_importances

		avg, std = [], []
		for i in range(5):
			avg.append(np.mean(fi[:,i]))
			std.append(np.std(fi[:,i]))

		avg = np.array(avg)
		std = np.array(std)
		labels = np.array(labels)

		sort_indices = np.argsort(avg)

		ind = np.arange(len(labels))

		#pdb.set_trace()

		fig, ax = plt.figure(figsize=(11,8))
		rects1 = ax.bar(ind, avg[sort_indices], color='red', 
						yerr=std[sort_indices], ecolor='black', align='center')

		ax.set_ylabel('Feature Importance')
		ax.set_xticks(ind)
		ax.set_xticklabels(labels[sort_indices])
		ax.set_ylim(0, 0.45)
		ax.set_yticks([0., .1, .2, .3, .4])

		plt.savefig('RF_feature_importance_4paper.pdf')
		plt.show()

		#pdb.set_trace()



	if make_roc_curves:
		candidateFileList = mid_sim.fetchFileList(kind='candidate')
		"""
		# SWAP situation at ~30 days into simulation
		candidates1 = mid_sim.fetchCatalog(candidateFileList[30])
		rejected1 = mid_sim.fetchCatalog(mid_sim.rejectedFileList[30])
		swap_subjects1 = np.concatenate([candidates1, rejected1])
		subjects1 = join(gz2_metadata, swap_subjects1, keys='zooid')

		# SWAP situation at ~60 days into simualtion
		candidates2 = mid_sim.fetchCatalog(candidateFileList[60])
		rejected2 = mid_sim.fetchCatalog(mid_sim.rejectedFileList[60])
		swap_subjects2 = np.concatenate([candidates2, rejected2])
		subjects2 = join(gz2_metadata, swap_subjects2, keys='zooid')
		"""
		# SWAP situation at the end of the siulation
		candidates3 = mid_sim.fetchCatalog(candidateFileList[-1])
		rejected3 = mid_sim.fetchCatalog(mid_sim.rejectedFileList[-1])
		swap_subjects3 = np.concatenate([candidates3, rejected3])
		subjects3 = join(gz2_metadata, swap_subjects3, keys='zooid')

		subject_sets = [subjects3]#subjects1, subjects2, 

		plot_roc_curve(subject_sets, smooth_or_not=False, gz_kind='raw_combo', swap=True, outname=None)



	""" MAKE SWAP VARIATIONS PLOT(S) """
	if make_swap_variations_plot:

		#"""
		# Load up simulations varying subject PRIOR
		# -------------------------------------------------------
		low_p = 'sup_PLPD5_p2_flipfeature2_norand'
		high_p = 'sup_PLPD5_p8_flipfeature2_norand'
		p35 = 'sup_PLPD5_p35_flipfeature2_norand'

		low_p_eval2 = Table.read('tertiary_simulation_output/GZX_evaluation_{0}.txt'.format(low_p+'_raw_combo'), format='ascii')
		high_p_eval2 = Table.read('tertiary_simulation_output/GZX_evaluation_{0}.txt'.format(high_p+'_raw_combo'), format='ascii')
		#p35_eval2 = Table.read('tertiary_simulation_output/GZX_evaluation_{0}.txt'.format(p35+'_raw_combo'), format='ascii')

		low_p_sim = Simulation(config='configfiles/update_sup_PLPD5_p2_flipfeature2_norand.config',
							   directory='sims_SWAP/S_PLPD5_p2_ff_norand/',
							   variety='feat_or_not')

		high_p_sim = Simulation(config='configfiles/update_sup_PLPD5_p8_flipfeature2_norand.config',
								directory='sims_SWAP/S_PLPD5_p8_ff_norand/',
								variety='feat_or_not')

		#p35_sim = Simulation(config='configfiles/update_sup_PLPD5_p35_flipfeature2_norand.config',
		#					directory='sims_SWAP/S_PLPD5_p35_ff_norand/',
		#					variety='feat_or_not')

		#"""
		# Load up simulations for varying user PL/PD
		# -------------------------------------------------------
		low_plpd = 'sup_PLPD4_p5_flipfeature2_norand'
		high_plpd = 'sup_PLPD6_p5_flipfeature2_norand'

		low_plpd_eval2 = Table.read('tertiary_simulation_output/GZX_evaluation_{0}.txt'.format(low_plpd+'_raw_combo'), format='ascii')
		high_plpd_eval2 = Table.read('tertiary_simulation_output/GZX_evaluation_{0}.txt'.format(high_plpd+'_raw_combo'), format='ascii')

		low_plpd_sim = Simulation(config='configfiles/update_sup_PLPD4_p5_flipfeature2_norand.config',
								  directory='sims_SWAP/S_PLPD4_p5_ff_norand/',
								  variety='feat_or_not')

		high_plpd_sim = Simulation(config='configfiles/update_sup_PLPD6_p5_flipfeature2_norand.config',
								   directory='sims_SWAP/S_PLPD6_p5_ff_norand/',
								   variety='feat_or_not')


		#"""
		# VARY PRIOR
		fig = plt.figure(figsize=(11,16))
		plt.rc('text', usetex=True)

		gs = gridspec.GridSpec(2,1)
		gs.update(wspace=0.05, hspace=0.01)

		ax = fig.add_subplot(gs[0])
		plot_GZX_evaluation_spread(92, low_p_eval2, mid_eval2, high_p_eval2,
							 	   outfile='compare_PLPD_4paper', ax=ax)

		ax2 = fig.add_subplot(gs[1])
		plot_GZX_cumulative_retirement_spread(92, low_p_sim, mid_sim, high_p_sim,
											  gz2_cum_sub_retired, 
											  outfile='compare_prior_4paper', ax=ax2)
		
		fig.suptitle(r'$0.1 \le \mathrm{Subject~Prior} \le 0.8$', fontsize=30)    

		gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
		plt.savefig('GZX_eval_and_retirement_prior_spread_4paper_v2.pdf')
		plt.show()
		plt.close()
		#"""
		# -----------------------------------------------------------
		# VARY PLPD
		fig = plt.figure(figsize=(11,16))
		plt.rc('text', usetex=True)

		gs = gridspec.GridSpec(2,1)
		gs.update(wspace=0.01, hspace=0.01)


		ax = fig.add_subplot(gs[0])
		plot_GZX_evaluation_spread(92, low_plpd_eval2, mid_eval2, high_plpd_eval2, 
							 	   outfile='compare_PLPD_4paper', ax=ax)

		ax2 = fig.add_subplot(gs[1])
		plot_GZX_cumulative_retirement_spread(92, low_plpd_sim, mid_sim, high_plpd_sim, 
											  gz2_cum_sub_retired, 
											  outfile='compare_prior_4paper', ax=ax2)


		fig.suptitle(r'$(0.4, 0.4) \le \mathrm{Confusion~Matrix} \le (0.6, 0.6)$', fontsize=30)    

		gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
		plt.savefig('GZX_eval_and_retirement_PLPD_spread_4paper_v2.pdf')
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