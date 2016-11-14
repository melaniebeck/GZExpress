from __future__ import division

import subprocess, sys, os, pdb
from astropy.table import Table, join, vstack
import numpy as np
import swap
from find_indices import find_indices
#from .errors import SimulationConfigNotSpecified

class Simulation(object):

	def __init__(self, days=None, config=None, variety='smooth_or_not', directory='.'):
		
		if config is None:
			print "Shit be fucked"
			#raise SimulationConfigNotSpecified("A simulation config file must "\
			#								   "be specified to create a simulation.")
		self.config = config
		self.parameters = self.fetchParameters()
		self.name = self.parameters['survey']
		self.directory = directory
		self.variety = variety

		self.detectedFileList = self.fetchFileList(kind='detected')
		self.rejectedFileList = self.fetchFileList(kind='rejected')
		self.retiredFileList = self.fetchFileList(kind='retired')

		self.days = self.fetchNumberOfDays(days)

		self.size = self.fetchSimSize()

		self._bureau = None
		self._collection = None
		self._subjectMetadata = None


	@property
	def bureau(self):
		if self._bureau is None:
			self.bureau = swap.read_pickle(self.parameters['bureaufile'], 'bureau')
		return self._bureau

	@property
	def collection(self):
		if self._collection is None:
			self._collection = swap.read_pickle(self.parameters['samplefile'], 'collection')			
		return self._collection

	@property
	def subjectMetadata(self):
		if self._subjectMetadata is None:
			self._subjectMetadata = swap.read_pickle(self.parameters['metadatafile'], 
													 'metadata').subjects
		return self._subjectMetadata
	
	
	

	def fetchNumberOfDays(self, days):
		''' Settle on the number of days in this simulation... '''

		# If user provides NO value, assume they want the entire simulation 
		# (each file represents one "day", so count the number of files)
		option1_days = int(subprocess.check_output("find %s/%s_*/ -maxdepth 1 "\
												   " -type d -print | wc -l"\
												   %(self.directory, self.name), 
												   shell=True))

		# Another way to isolate the number of days is to see which files actually exist
		if len(self.detectedFileList) == len(self.rejectedFileList):
			# Hopefully the number of detectedfile == number of rejectedfiles (1 / day)
			option2_days = len(self.detectedFileList)
		else:
			# If it doesn't, take the minimum so that we don't try to over access
			option2_days = min(len(self.detectedFileList), len(self.rejectedFileList))

		# If the user has provided nothing at all... 
		if days is None:
			# take the minimum of all the options above
			sim_days = min(option1_days, option2_days)

		# If they did provide a value but it's TOO LARGE
		elif days > option2_days:
			sim_days = option2_days	

		else:
			sim_days = days

		return sim_days

	def fetchSimSize(self):
		''' compute size of simulation at the day requested by user '''
		if self.days != None:
			all_detected = self.fetchCatalog(self.detectedFileList[self.days-1])
			all_rejected = self.fetchCatalog(self.rejectedFileList[self.days-1])
			return len(all_detected) + len(all_rejected)


	def fetchParameters(self):
		''' At it's base, a simulation is an "instance" of SWAP -- open config '''
		p = swap.Configuration(self.config)
		return p.parameters


	def fetchFileList(self, kind):
		''' Fetch the list of catalogs which contain the # of subjects/day '''
		try:
			cmd = "ls %s/%s_*/*%s_catalog.txt"%(self.directory, self.name, kind)
			cmdout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
			filelist = cmdout.stdout.read().splitlines()
		except:
			print "No '%s' files found! Aborting...\n"%kind
			sys.exit()
		return filelist


	@staticmethod
	def fetchCatalog(filename):
		try:
			catalog = Table.read(filename, format='ascii')
		except IOError:
			print "Simulation: %s catalog could not be opened."%filename
		else:
			return catalog


	def computeCumulativeRetiredSubjects(self, kind='detected'):

		if kind == 'detected':
			filelist = self.detectedFileList
		elif kind == 'rejected':
			filelist = self.rejectedFileList
		elif kind == 'both' or kind == 'retired':
			filelist = self.retiredFileList
		else:
			# Raise exception; for now, print error
			print "{0} is not available as a filelist".format(kind)
			sys.exit()

		cumulative_retired_subjects = []
		for filename in filelist:
			with open(filename,'rb') as f:
				dat = f.read().splitlines()
				cumulative_retired_subjects.append(len(dat))

		return np.array(cumulative_retired_subjects)


	def fetchRetiredSubjectsByDate(self):

		print "Simulation: generating full catalog of retired subjects, sorted by day retired"

		for day, filename in enumerate(self.retiredFileList):

			# Cat is all subjects retired on *THIS* day (includes previous days)
			current_cat = self.fetchCatalog(filename)
			current_cat['day'] = day

			if day != 0:
				# New set elements in CURRENT_CAT but not (yet) in FULL_CAT
				new_subjectIDs = set(current_cat['zooid']) - set(full_cat['zooid'])

				# Find INDICES where the new subject IDs can be found in CURRENT_CAT
				ind = find_indices(current_cat['zooid'], np.array(list(new_subjectIDs)))

				new_cat = current_cat[ind]

				# Add New Subjects to FULL_CAT
				full_cat = vstack([full_cat, new_cat])

			else:
				full_cat = current_cat

		return full_cat


	# Plotting functions for data entirely internal to Simulation
	# -------------------------------------------------------------------------
	def plot_clicks_till_retired(self, axes, plot_combined=False):

		detected = self.fetchCatalog(self.detectedFileList[-1])	
		rejected = self.fetchCatalog(self.rejectedFileList[-1])

		num_detected = detected['Nclass']
		num_rejected = rejected['Nclass']
		num_combined = np.concatenate([num_rejected, num_detected])

		if self.variety == 'smooth_or_not':
			labels = ["'Smooth' (%i)"%len(num_detected),
					  "'Not' (%i)"%len(num_rejected)]
			colors = ['orange','yellow']
			edgecolors = ['darkorange','gold']
		else:
			labels = ["'Featured' (%i)"%len(num_detected),
					  "'Not' (%i)"%len(num_rejected)]
			colors = ['yellow', 'orange']
			edgecolors = ['gold','darkorange']

		# Plot the number of clicks till retirement for the detected-
		# and rejected-labeled subjects

		bins = np.arange(0, np.max(num_combined)+1, 1)


		if plot_combined:
			weights = np.ones_like(num_combined, dtype='float64')/len(num_combined)

			det = np.concatenate([num_detected, np.full(len(num_rejected), -1, dtype='int64')])
			rej = np.concatenate([num_rejected, np.full(len(num_detected), -1, dtype='int64')])
			
			axes.hist(num_combined, weights=weights, bins=bins, range=(0,50), 
				  	  histtype='stepfilled', alpha=0.5, color='lightsteelblue', 
				  	  edgecolor='steelblue', lw=2, label='SWAP')#'All Retired (%i)'%len(num_combined)

			axes.hist(det, weights=weights, bins=bins, range=(0,50), histtype='step',
					  color='steelblue', hatch='/', lw=2, label="'Featured'")
			axes.hist(rej, weights=weights, bins=bins, range=(0,50), histtype='step',
					  color='steelblue', hatch='\\', lw=2, label="'Not'")

		else:
			axes.hist(num_detected, normed=True, bins=bins, range=(0,50), 
					  histtype='stepfilled', alpha=0.75, color=colors[0], 
					  edgecolor=edgecolors[0], linewidth=2, label=labels[0])

			axes.hist(num_rejected, normed=True, bins=bins, range=(0,50), 
					  histtype='stepfilled', alpha=0.75, color=colors[1], 
					  edgecolor=edgecolors[1], linewidth=2, label=labels[1])


		axes.set_ylim(0,.2)
		axes.set_xlim(0,60.)
		axes.set_xlabel('Classifications till retirement')
		axes.set_ylabel('Frequency')
		axes.legend(loc='best',frameon=False)

		return 


	def plot_cumulative_retired_subjects(self, axes, plot_combined=False, 
										 fill_between=False):

		import matplotlib.ticker as ticker

		dates = np.arange(self.days)

		detected = self.computeCumulativeRetiredSubjects(kind='detected')
		rejected = self.computeCumulativeRetiredSubjects(kind='rejected')

		if plot_combined:
			if fill_between:
				axes.fill_between(dates, detected+rejected, color='lightsteelblue',
							  	  edgecolor='steelblue', lw=2, alpha=0.5, label='SWAP')
			else:
				axes.plot(dates, detected+rejected, color='steelblue', lw=3, label='SWAP')

		else:
			if self.variety == 'smooth_or_not':
				labels = ["GZX:'Smooth'", "GZX:'Not'"]
				colors = ['orange','yellow']
				edgecolors = ['darkorange','gold']
			else:
				labels = ["GZX:'Featured'", "GZX:'Not'"]
				colors = ['yellow', 'orange']
				edgecolors = ['gold','darkorange']

			axes.fill_between(dates, rejected, detected+rejected,
						  color=colors[0], edgecolor=edgecolors[0],
						  lw=2, alpha=0.5, label=labels[0])

			axes.fill_between(dates, rejected, color=colors[1], lw=2,
						  edgecolor=edgecolors[1], alpha=0.5, 
						  label=labels[1])


		axes.set_xlim(0, self.days-1)
		axes.set_ylim(0,300000)

		x = detected+rejected
		scale_y = 1e3
		ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
		axes.yaxis.set_major_formatter(ticks_y)

		yticks = np.arange(0,360000,60000)
		axes.set_yticks(yticks)

		axes.set_xlabel('Days in GZ2 Project')
		axes.set_ylabel(r'Cumulative retired subjects  [$10^3$]')
	
		axes.legend(loc='upper left',frameon=False)

		return axes

	"""
	col1, col2, col3, col4, col5 = [], [], [], [], []
	for day, filename in enumerate(mid_sim.retiredFileList):

		print day
		print filename
		print ""

		if day != 0:
			# Cat is all subjects retired on *THIS* day (includes previous days)
			with open(filename, 'rb') as f:
				header = f.readline()

				#dat = f.read().splitlines()
				for line in f:
					line = line.strip()
					columns = line.split()
					subjectID = int(columns[0])

					if subjectID not in full_cat_IDs:
						col1.append(columns[0])
						col2.append(float(columns[1]))
						col3.append(int(columns[2]))
						col4.append(str(columns[3]))
						col5.append(day)

						full_cat_IDs.append(subjectID)

		else:
			full_cat = mid_sim.fetchCatalog(filename)
			full_cat_IDs = list(full_cat['zooid'])
			full_cat['day'] = 0


	pdb.set_trace()
	new = Table(data=(col1, col2, col3, col4, col5), 
				names=('zooid', 'P', 'Nclass', 'image', 'day'))
	full_cat = vstack([full_cat, new])
	"""