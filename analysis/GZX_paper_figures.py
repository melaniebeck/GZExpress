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

from find_indices import find_indices



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

		# Load up the SWAP Simulation AGENT BUREAU
		bureau = swap.read_pickle('GZ2_sup_PLPD5_p5_flipfeature2b_bureau.pickle', 'bureau')
		plot_user_probabilities(bureau, 200)


	""" MAKE BASELINE SIMULATION PLOT """
	if make_baseline_simulation_plot:

		# BASELINE fig requires BASELINE Simulation, 
		#						evaluation output for that sim,
		#						cumulative retirement for GZ2
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


	""" MORPH DISTRIBUTIONS """
	if make_morph_distributions_plot:

		# Plotting FEAT vs NOT, FALSE POS & FALSE NEGs, RETIRED vs NOT RETIRED
		# to do all that, need files that were created.... GZX_SWAP_eval?
		outfile = 'GZ2_sup_PLPD5_p5_flipfeature2b_RF_accuracy_redo_raw_combo'
		machine_retired = Table.read('{}_machine_retired_subjects.fits'.format(outfile))
		machine_not_retired = Table.read('{}_machine_not_retired_subjects.fits'.format(outfile))

		plot_morph_params_1D(machine_retired, machine_not_retired, gz2_metadata, outfile)


	""" MAKE SWAP GETS IT WRONG PLOT """
	if make_swap_gets_it_wrong_plot:

		# Compare SWAP-retired subjects to various parameters in the GZ2 Main Catalog
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

		# Requires the Vote Distributions for GZ2 and those from the Simulation
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