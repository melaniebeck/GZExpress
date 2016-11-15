## GZX combines SWAP with a machine classifier and has been tested through simulations on GZ2 classifications
Here we briefly describe some of the modifications of SWAP, the machine classifying framework, and the simulations we perform. 


### Getting Started
GZ2 classifications are stored in a MySQL database. `swap/mysqldb.py` was created to pull classifications from this database instead of the Space Warps `swap/mongodb.py`.
In order to reduce the time of each SQL query, tables in the GZ2 db were joined before running SWAP. The script for the particular arrangement can be found in `prepare_gz2sql.py`. 


Additionally, a Storage Locker pickle is required for both the SWAP and Machine components of GZX. This allows SWAP and the Machine to communicate. Each simulation generates its own version of this file through `swap/storage.py` which, in turn, requires `metadata_ground_truth_labels.fits` The latter is created only one time with `create_GZ2_truth_label_catalog.py`. [This should be a function for `galaxyzoo2.py` Singleton]

Specifically, this file contains 
	* the feature vectors which the machines train on (morphology indicators measured from the pixel values of the original FITS files for the galaxy images), 
	* the truth labels from the GZ2 published data, 
	* a tag that specifies that galaxy as one of the following samples:
		* *train* -- Retired by SWAP; training sample for Machine
		* *valid* -- Retired by SWAP; expertly classified; machine validation sample
		* *test*  -- Not yet retired by either SWAP or Machine; the Machine's test sample
		* *mclas* -- Retired by Machine 
	All subjects start as *test* except for the expertly-classified/validation sample


### SWAP Workflow
`SWAP.py` has been modified to run on a "daily" basis. A config parameter called "increment" controls the timestep in units of days. Only classifications with timestamps within the increment are collected and processed.   

Subjects which cross either the acceptance or rejection thresholds have their tag in the storage locker modifed to *train*

For each night, SWAP creates a slew of output files (report is always True): 

    SURVEY_Y-D-H_00:00:00_candidate_catalog.txt     # any subject which has been classified
    SURVEY_Y-D-H_00:00:00_candidates.txt        
    SURVEY_Y-D-H_00:00:00_detected_catalog.txt      # any subject which has crossed the acceptance threshold
    SURVEY_Y-D-H_00:00:00_dud_catalog.txt       
    SURVEY_Y-D-H_00:00:00_retired_catalog.txt       # any subject which has crossed the rejectance threshold
    SURVEY_Y-D-H_00:00:00_sim_catalog.txt
    SURVEY_Y-D-H_00:00:00_training_false_negatives.txt
    SURVEY_Y-D-H_00:00:00_training_false_positives.txt
    SURVEY_Y-D-H_00:00:00_training_true_positives.txt

---

### Machine Workflow
`MachineClassifier.py` also reads the config file in similar fashion to SWAP. The config file now contains several parameters which control how the machine trains. 

Here's what it currently does: 
 * selects out the *valid* sample and the *train* sample based on the tags in the Storage Locker
 * performs cross validation with the training sample to determine appropriate machine hyperparameters
 * creates an **agent** for the machine which tracks the machine's training and validation history 
 * Once the agent has determined the machine is suitably trained, the machine classifier is applied to the *test* sample 

---

### SWAPSHOP and MachineShop
`SWAPSHOP.py` was created to run SWAP in batch mode. This was used to create several Simulations for a variety of input SWAP parameters (individual simulations can be managed with the `simulation.py` class). Similarly, `MachineShop.py` performs batch runs which combine the SWAP Simulation output with the Machine Classifier. 

This is an OFFLINE way to run GZX. It uses the output SWAP files mentioned above as input (along with the Storage Locker) to determine which subjects are classified by the machine each night (instead of SWAP). 

The ONLINE version of GZX hasn't yet had all the kinks worked out. The basic structure of the ONLINE version will first run SWAP.py and then MachineClassifier.py directly afterwards. Both will modify the collection pickle instead of (or in addition to) the Storage Locker. 


### SWAP-only Simulations

A slew of simulations have been run to explore the parameter space of SWAP. Most important parameters are the intitial volunteer confusion matrix,  the subject prior probability, and acceptance and rejectance thresholds. 

Simulations that we have performed:

|Date Started|Sample|Train Strategy|Subject Labels|Configuration|PL, PD|Prior|Thresholds|Last GZ2 Day|RunName|NOTES|
|-----|----|----|----|----|----|----|----|----|----|
|6/30/2016|expertsample|trainfirst|S or N|S&U|0.5, 0.5|0.3|0.004, 0.99|2/25/2009|GZ2_sup_unsup_0.5_trainfirst_standard2|Using "Expert_label"|Can't run code using "Nair_label" on one and "Expert_label" on the other|
|6/30/2016|expertsample|trainfirst|S or N|S only|0.5, 0.5|0.3|0.004, 0.99|2/25/2009|GZ2_sup_0.5_trainfirst_standard2|Using "Expert_label"|because the code is all in the same directory. These have to be done in order.|
|6/29/2016|expertsample|trainfirst|F or N|S&U|0.5, 0.5|0.3|0.004, 0.99|2/25/2009|GZ2_sup_unsup_0.5_trainfirst_flipfeature|Using "Nair_label" |
|6/29/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.3|0.004, 0.99|2/25/2009|GZ2_sup_0.5_trainfirst_flipfeature|Using "Nair_label" |
|6/29/2016|expertsample|trainfirst|F or N|S&U|0.5, 0.5|0.3|0.004, 0.99|2/25/2009|GZ2_sup_unsup_0.5_trainfirst_flipfeature2|Using "Expert_label"|
|6/29/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.3|0.004, 0.99|5/14/2009|GZ2_sup_0.5_trainfirst_flipfeature2|Using "Expert_label"|This run got fucked -- pickle files wiped out. Nightly output exists in S_PLPD5_flipfeature2b/. Re-ran the full sim; in S_PLPD5_flipfeature2b_second/ |
|7/7/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.5|0.004, 0.99|5/14/2009|GZ2_sup_PLPD5_p5_flipfeature2|Using "Expert_label"|
|7/7/2016|expertsample|trainfirst|S or N|S only|0.5, 0.5|0.5|0.004, 0.99|2/25/2009|GZ2_sup_PLPD5_p5_standard2|Using "Expert_label"|
|7/15/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.5|0.004, 0.99|2/25/2009|GZ2_sup_PLPD5_p5_flipfeature2b|Treating Stars&Artifacts as Featured -- always |In S/N runs, Star/Artifact was lumped in with "Not" but in previous F/N runs 
|7/15/2016|expertsample|trainfirst|F or N|S&U|0.5, 0.5|0.5|0.004, 0.99|2/25/2009|GZ2_sup_PLPD5_p5_flipfeature2b|Treating Stars&Artifacts as Featured -- always|it was lumped in with Smooth. These two runs -- Star/Artifact STAYS with F
|7/26/2016|expertsample|trainfirst|S or N|S only|0.5, 0.5|0.5|0.004, 0.99|5/14/2009|in folder S_PLPD5_standard2_norandom2/|Turned off the "random realizations"|Where did that occur?? I can't remember where this was in the code....
|7/26/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.5|0.004, 0.99|5/14/2009|in folder S_PLPD5_flipfeature2_norandom2/|Turned off the "random realizations"|
|8/18/2016|expertsample|trainfirst|F or N|S only|0.4, 0.4|0.5|0.004, 0.99|5/14/2009|GZ2_sup_PLPD4_p5_flipfeature2_norand|Change PL,PD|From here on down we're going with F/N, S only, NO random realizations!|
|8/18/2016|expertsample|trainfirst|F or N|S only|0.6, 0.6|0.5|0.004, 0.99|5/14/2009|GZ2_sup_PLPD6_p5_flipfeature2_norand|Change PL,PD|
|8/19/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.8|0.004, 0.99|5/14/2009|GZ2_sup_PLPD5_p8_flipfeature2_norand|Change Prior|
|8/19/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.2|0.004, 0.99|5/14/2009|GZ2_sup_PLPD5_p2_flipfeature2_norand|Change Prior|
|9/7/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.01|0.004, 0.99|5/14/2009|GZ2_sup_PLPD5_p01_flipfeature2_norand|Change Prior|
|9/9/2016|expertsample|trainfirst|F or N|S only|0.5, 0.5|0.35|0.004, 0.99|5/14/2009|GZ2_sup_PLPD5_p35_flipfeature2_norand|Change Prior|




### What needs to be done [really should create issues for these...]:
* Determine how the Machine retires (threshold); flag retired galaxies so they are no longer processed through SWAP (This is being developed/explored in `explore_machine.py`)
* Once the above is implemented, run the first LIVE test of SWAP + MachineClassifier (tests so far have had MC in 'offline' mode.
* Try: Instead of using candidate and rejected catalogs - try detected and rejected catalogs? Within a few days there are so many classifications being processed by SWAP that the training sample becomes huge and the test sample miserably small.
* Try: Setting aside a strictly fixed test sample of XXX subjects (if they end up being classified, don't use them in the machine classifier)
* Try:  Break training and test samples up by redshift? magnitude? color? (test this separately?)


