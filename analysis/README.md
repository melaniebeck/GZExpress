## GZX combines SWAP with a machine classifier and has been tested through simulations on GZ2 classifications
Here we briefly describe some of the modifications of SWAP, the machine classifying framework, and the simulations we perform. 


### Getting Started
GZ2 classifications are stored in a MySQL database. `swap/mysqldb.py` was created to pull classifications from this database instead of the Space Warps `swap/mongodb.py`.
In order to reduce the time of each SQL query, tables in the GZ2 db were joined before running SWAP. The script for the particular arrangement can be found in `prepare_gz2sql.py`. 


Additionally, a Storage/metadata pickle is required for both the SWAP and Machine components of GZX. This allows SWAP and the Machine to communicate. Each simulation generates its own version of this file through `swap/storage.py` which, in turn, requires `metadata_ground_truth_labels.fits` The latter is created only one time with `create_GZ2_truth_label_catalog.py`.

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
 * reads `update.config`
 * reads in the metadata pickle
 * selects out the *valid* sample and the *train* sample based on the tags in the metadata pickle
 * performs cross validation with the training sample to determine appropriate machine hyperparameters
 * creates an **agent** for the machine which tracks the machine's training and validation history 
 * Once the agent has determined the machine is suitably trained, the machine classifier is applied to the *test* sample 

---

### Machine Shop
In parallel to `SWAPSHOP.py`, `MachineShop.py` runs `MachineClassifier.py` on timesteps. 


### Experiments to run
A slew of simulations (with and without MC) need to be run to explore the parameter space of SWAP. Parameters of interest include the acceptance and rejectance thresholds, the prior, and the initialPL and initialPD of user-agents
* Are # classifications relatively immune to initialPL and initialPD values? Vary PL/PD between .55 - .8
* Does it matter if PL and PD are the same or different?
* How drastically do classifications change when rejectance/acceptance thresholds change?
* What affect does the initial prior have? (currently set at 0.3)

Work should also be done on other questions in the GZ2 decision tree. 
* Bar or Not?  (task 3)
* Spiral arms or Not? (task 4)
* Bulge or Not? (task 5)
* Edge on or Not? (task 2)



### What needs to be done [really should create issues for these...]:
* Determine how the Machine retires (threshold); flag retired galaxies so they are no longer processed through SWAP (This is being developed/explored in `explore_machine.py`)
* Once the above is implemented, run the first LIVE test of SWAP + MachineClassifier (tests so far have had MC in 'offline' mode.
* Try: Instead of using candidate and rejected catalogs - try detected and rejected catalogs? Within a few days there are so many classifications being processed by SWAP that the training sample becomes huge and the test sample miserably small.
* Try: Setting aside a strictly fixed test sample of XXX subjects (if they end up being classified, don't use them in the machine classifier)
* Try:  Break training and test samples up by redshift? magnitude? color? (test this separately?)


