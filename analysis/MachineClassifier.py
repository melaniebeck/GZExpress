from __future__ import division

from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import classification_report
import sklearn.metrics as mtrx
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier as RF

import swap
import machine_utils as ml
#import metrics as mtrx
from metrics import compute_binary_metrics

from optparse import OptionParser
from astropy.table import Table
import pdb
import numpy as np
import datetime as dt 
import os, subprocess, sys
import cPickle

'''
Workflow:
   access morphology database
   accept labels/data for training
   accept labels/data for testing
   "whiten" data (normalize)
   {reduce dimensions} (optional)s
   train the machine classifier
   run machine classifier on test sample
'''

def MachineClassifier(options, args):
    """
    NAME
        MachineClassifier.py

    PURPOSE
        Machine learning component of Galaxy Zoo Express

        Read in a training sample generated by human users (which have 
        previously been analyzed by SWAP).
        Learn on the training sample and moniter progress. 
        Once "fully trained", apply learned model to test sample. 

    COMMENTS
        Lots I'm sure. 

    FLAGS
        -h            Print this message
        -c            config file name 
    """


    #-----------------------------------------------------------------------    
    #                 LOAD CONFIG FILE PARAMETERS  
    #-----------------------------------------------------------------------
    # Check for config file in array args:
    if (len(args) >= 1) or (options.configfile):
        if args: config = args[0]
        elif options.configfile: config = options.configfile
        print swap.doubledashedline
        print swap.ML_hello
        print swap.doubledashedline
        print "ML: taking instructions from",config
    else:
        print MachineClassifier.__doc__
        return

    machine_sim_directory = 'sims_Machine/redo_with_circular_morphs/'

    tonights = swap.Configuration(config)
    
    # Read the pickled random state file
    random_file = open(tonights.parameters['random_file'],"r");
    random_state = cPickle.load(random_file);
    random_file.close();
    np.random.set_state(random_state)

    time = tonights.parameters['start']

    # Get the machine threshold (to make retirement decisions)
    swap_thresholds = {}
    swap_thresholds['detection'] = tonights.parameters['detection_threshold']  
    swap_thresholds['rejection'] = tonights.parameters['rejection_threshold']
    threshold = tonights.parameters['machine_threshold']
    prior = tonights.parameters['prior']

    # Get list of evaluation metrics and criteria   
    eval_metrics = tonights.parameters['evaluation_metrics']
    
    # How much cross-validation should we do? 
    cv = tonights.parameters['cross_validation']

    survey = tonights.parameters['survey']

    # To generate training labels based on the subject probability, 
    # we need to know what should be considered the positive label: 
    # i.e., GZ2 has labels (in metadatafile) Smooth = 1, Feat = 0
    # Doing a Smooth or Not run, the positive label is 1
    # Doing a Featured or Not run, the positive label is 0
    pos_label = tonights.parameters['positive_label']

    #----------------------------------------------------------------------
    # read in the metadata for all subjects
    storage = swap.read_pickle(tonights.parameters['metadatafile'], 'metadata')

    # 11TH HOUR QUICK FIX CUZ I FUCKED UP. MB 10/27/16
    if 'GZ2_raw_combo' not in storage.subjects.colnames:
        gz2_metadata = Table.read('metadata_ground_truth_labels.fits')
        storage.subjects['GZ2_raw_combo'] = gz2_metadata['GZ2_raw_combo']
        swap.write_pickle(storage, tonights.parameters['metadatafile'])

    subjects = storage.subjects

    #----------------------------------------------------------------------
    # read in the PROJECT COLLECTION -- (shared between SWAP/Machine)
    #sample = swap.read_pickle(tonights.parameters['samplefile'],'collection')

    # read in or create the ML bureau for machine agents (history for Machines)
    MLbureau = swap.read_pickle(tonights.parameters['MLbureaufile'],'bureau')



    #-----------------------------------------------------------------------    
    #                 FETCH TRAINING & VALIDATION SAMPLES  
    #-----------------------------------------------------------------------
    train_sample = storage.fetch_subsample(sample_type='train',
                                           class_label='GZ2_raw_combo')
    """ Notes about the training sample:
    # this will select only those which have my morphology measured for them
    # AND which have "ground truth" according to GZ2
    # Eventually we could open this up to include the ~10k that aren't in the 
    # GZ Main Sample but I think, for now, we should reduce ourselves to this
    # stricter sample so that we always have back-up "truth" for each galaxy.
    """

    try:
        train_meta, train_features = ml.extract_features(train_sample, 
                                        keys=['M20_corr', 'C_corr', 'E', 'A_corr', 'G_corr'])
        original_length = len(train_meta)

    except TypeError:
        print "ML: can't extract features from subsample."
        print "ML: Exiting MachineClassifier.py"
        sys.exit()

    else:
        # TODO: consider making this part of SWAP's duties? 
        # 5/18/16: Only use those subjects which are no longer on the prior
        off_the_fence = np.where(train_meta['SWAP_prob']!=prior)
        train_meta = train_meta[off_the_fence]
        train_features = train_features[off_the_fence]
        train_labels = np.array([pos_label if p > prior else 1-pos_label 
                                 for p in train_meta['SWAP_prob']])


        shortened_length = len(train_meta)
        print "ML: found a training sample of %i subjects"%shortened_length
        removed = original_length - shortened_length
        print "ML: %i subjects removed to create balanced training sample"%removed
    

    valid_sample = storage.fetch_subsample(sample_type='valid',
                                           class_label='Expert_label')
    try:
        valid_meta, valid_features = ml.extract_features(valid_sample,
                                        keys=['M20_corr', 'C_corr', 'E', 'A_corr', 'G_corr'])
    except:
        print "ML: there are no subjects with the label 'valid'!"
    else:
        valid_labels = valid_meta['Expert_label'].filled()
        print "ML: found a validation sample of %i subjects"%len(valid_meta)

    # ---------------------------------------------------------------------
    # Require a minimum size training sample [Be reasonable, my good man!]
    # ---------------------------------------------------------------------
    if len(train_sample) < 10000: 
        print "ML: training sample is too small to be worth anything."
        print "ML: Exiting MachineClassifier.py"
        sys.exit()
        
    else:
        print "ML: training sample is large enough to give it a shot."

        # TODO: LOOP THROUGH DIFFERENT MACHINES? 
        # 5/12/16 -- no... need to make THIS a class and create multiple 
        #            instances? Each one can be passed an instance of a machine?

        # Machine can be trained to optimize different metrics
        # (ACC, completeness, purity, etc. Have a list of acceptable ones.)
        # Minimize a Loss function. 
        for metric in eval_metrics:
        
            # REGISTER Machine Classifier
            # Construct machine name --> Machine+Metric
            machine = 'RF'
            Name = machine+'_'+metric
        
            # register an Agent for this Machine
            try: 
                test = MLbureau.member[Name]
            except: 
                MLbureau.member[Name] = swap.Agent_ML(Name, metric)
                
            MLagent = MLbureau.member[Name]

            #---------------------------------------------------------------    
            #     TRAIN THE MACHINE; EVALUATE ON VALIDATION SAMPLE
            #---------------------------------------------------------------

            # Now we run the machine -- need cross validation on whatever size 
            # training sample we have .. 
        
            # Fixed until we build in other machine options
            # Need to dynamically determine appropriate parameters...

            #max_neighbors = get_max_neighbors(train_features, cv)
            #n_neighbors = np.arange(1, (cv-1)*max_neighbors/cv, 5, dtype=int)
            #params = {'n_neighbors':n_neighbors, 
            #          'weights':('uniform','distance')}

            num_features = train_features.shape[1]
        
            min_features = int(round(np.sqrt(num_features)))
            params = {'max_features':np.arange(min_features, num_features+1),
                      'max_depth':np.arange(2,16)}

            # Create the model 
            # for "estimator=XXX" all you need is an instance of a machine -- 
            # any scikit-learn machine will do. However, non-sklearn machines..
            # That will be a bit trickier! (i.e. Phil's conv-nets)
            general_model = GridSearchCV(estimator=RF(n_estimators=30), 
                                         param_grid=params, n_jobs=31,
                                         error_score=0, scoring=metric, cv=cv) 
            
            # Train the model -- k-fold cross validation is embedded
            print "ML: Searching the hyperparameter space for values that "\
                  "optimize the %s."%metric

            trained_model = general_model.fit(train_features, train_labels)
            MLagent.model = trained_model

            # Test accuracy (metric of choice) on validation sample
            score = trained_model.score(valid_features, valid_labels)

            ratio = np.sum(train_labels==pos_label) / len(train_labels)

            MLagent.record_training(model_described_by=
                                    trained_model.best_estimator_, 
                                    with_params=trained_model.best_params_, 
                                    trained_on=len(train_features), 
                                    with_ratio=ratio,
                                    at_time=time, 
                                    with_train_score=trained_model.best_score_,
                                    and_valid_score=trained_model.score(
                                        valid_features, valid_labels))

            valid_prob_thresh = trained_model.predict_proba(valid_features)[:,pos_label]
            fps, tps, thresh = mtrx.roc_curve(valid_labels,valid_prob_thresh, pos_label=pos_label)

            metric_list = compute_binary_metrics(fps, tps)
            ACC, TPR, FPR, FNR, TNR, PPV, FDR, FOR, NPV = metric_list
        
            MLagent.record_validation(accuracy=ACC, recall=TPR, precision=PPV,
                                      false_pos=FPR, completeness_f=TNR,
                                      contamination_f=NPV)
            
            #MLagent.plot_ROC()

            # ---------------------------------------------------------------
            # IF TRAINED MACHINE PREDICTS WELL ON VALIDATION ....
            # ---------------------------------------------------------------
            if MLagent.is_trained(metric) or MLagent.trained:
                print "ML: %s has successfully trained and will be applied "\
                      "to the test sample."%Name

                # Retrieve the test sample 
                test_sample = storage.fetch_subsample(sample_type='test',
                                                      class_label='GZ2_raw_combo')
                """ Notes on test sample:
                The test sample will, in real life, be those subjects for which
                we don't have an answer a priori. However, for now, this sample
                is how we will judge, in part, the performance of the overall
                method. As such, we only include those subjects which have 
                GZ2 labels in the Main Sample.
                """

                try:
                    test_meta, test_features = ml.extract_features(test_sample,
                                                keys=['M20_corr', 'C_corr', 'E', 'A_corr', 'G_corr'])
                except:
                    print "ML: there are no subjects with the label 'test'!"
                    print "ML: Either there is nothing more to do or there is a BIG mistake..."
                else:
                    print "ML: found test sample of %i subjects"%len(test_meta)

                #-----------------------------------------------------------    
                #                 APPLY MACHINE TO TEST SAMPLE
                #----------------------------------------------------------- 
                predictions = MLagent.model.predict(test_features)
                probabilities = MLagent.model.predict_proba(test_features)[:,pos_label]

                print "ML: %s has finished predicting labels for the test "\
                      "sample."%Name
                print "ML: Generating performance report on the test sample:"

                test_labels = test_meta['GZ2_raw_combo'].filled()
                print mtrx.classification_report(test_labels, predictions)

                test_accuracy = mtrx.accuracy_score(test_labels,predictions)
                test_precision = mtrx.precision_score(test_labels,predictions,pos_label=pos_label)
                test_recall = mtrx.recall_score(test_labels,predictions,pos_label=pos_label)

                MLagent.record_evaluation(accuracy_score=test_accuracy,
                                          precision_score=test_precision,
                                          recall_score=test_recall,
                                          at_time=time)
                
                # ----------------------------------------------------------
                # Save the predictions and probabilities to a new pickle

                test_meta['predictions'] = predictions
                test_meta['machine_probability'] = probabilities

                # If is hasn't been done already, save the current directory
                # ---------------------------------------------------------------------
                tonights.parameters['trunk'] = survey+'_'+tonights.parameters['start']
                # This is the standard directory... 
                #tonights.parameters['dir'] = os.getcwd()+'/'+tonights.parameters['trunk']

                # This is to put files into the sims_Machine/... directory. 
                tonights.parameters['dir'] = os.getcwd()
                filename=tonights.parameters['dir']+'/'+tonights.parameters['trunk']+'_'+Name+'.fits'
                test_meta.write(filename)

                count=0
                noSWAP=0
                for sub, pred, prob in zip(test_meta, predictions, probabilities):
                    
                    # IF MACHINE P >= THRESHOLD, INSERT INTO SWAP COLLECTION
                    # --------------------------------------------------------
                    if (prob >= threshold) or (1-prob >= threshold):

                        # Flip the set label in the metadata file -- 
                        #   don't want to use this as a training sample!
                        idx = np.where(subjects['asset_id'] == sub['asset_id'])
                        
                        storage.subjects['MLsample'][idx] = 'mclass'
                        storage.subjects['retired_date'][idx] = time
                        count+=1

                print "MC: Machine classifed {0} subjects with >= 90% confidence".format(count)
                print "ML: Of those, {0} had never been seen by SWAP".format(noSWAP)

    
    tonights.parameters['trunk'] = survey+'_'+tonights.parameters['start']
    tonights.parameters['dir'] = os.getcwd()
    if not os.path.exists(tonights.parameters['dir']):
        os.makedirs(tonights.parameters['dir'])


    # Repickle all the shits
    # -----------------------------------------------------------------------
    if tonights.parameters['repickle']:

        #new_samplefile = swap.get_new_filename(tonights.parameters,'collection')
        #print "ML: saving SWAP subjects to "+new_samplefile
        #swap.write_pickle(sample, new_samplefile)
        #tonights.parameters['samplefile'] = new_samplefile
        
        new_bureaufile=swap.get_new_filename(tonights.parameters,'bureau','ML')
        print "ML: saving MLbureau to "+new_bureaufile
        swap.write_pickle(MLbureau, new_bureaufile)
        tonights.parameters['MLbureaufile'] = new_bureaufile

        metadatafile = swap.get_new_filename(tonights.parameters,'metadata')
        print "ML: saving metadata to "+metadatafile
        swap.write_pickle(storage, metadatafile)
        tonights.parameters['metadatafile'] = metadatafile


    # UPDATE CONFIG FILE with pickle filenames, dir/trunk, and (maybe) new day
    # ----------------------------------------------------------------------
    configfile = config.replace('startup','update')

    # Random_file needs updating, else we always start from the same random
    # state when update.config is reread!
    random_file = open(tonights.parameters['random_file'],"w");
    random_state = np.random.get_state();
    cPickle.dump(random_state,random_file);
    random_file.close();

    swap.write_config(configfile, tonights.parameters)

    return



def get_max_neighbors(sample, cv_folds):
    # when performing cross validation using a KNN classifier, the number of 
    # nearest neighbors MUST be less than the sample size. 
    # Depending on how many folds one wishes their CV to compute, this changes
    # So! For the required number of folds, calculate the number of nearest 
    # neighbors which would be ONE less than the length of the sample size
    # once the FULL size of the sample has been split into num_folds groups
    # for cross validation. 
    # Furthermore, if we have a massively huge sample, we don't actually want 
    # to search the ENTIRE n_neighbors parameter space. Increasing the 
    # neighbors effectively smooths over the noise and we don't want to smooth 
    # TOO much. SO, return a capped value --
    # Minimum sample size = 100 right now, so max neighbors == 99

    cv_size = len(sample)*(1-1/cv_folds)-1
    max_neighbors = int(np.min([cv_size, 99]))
    return max_neighbors


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-c", dest="configfile", help="Name of config file")
    parser.add_option("-o", dest="offline", default=False, action='store_true',
                      help="Run in offline mode; e.g. on existing SWAP output.")
    parser.add_option("-v", "--verbose", action='store_true', dest="verbose")
    parser.add_option("-q", "--quiet", action="store_false", dest="verbose")

    (options, args) = parser.parse_args()

    MachineClassifier(options, args)

    #pdb.set_trace()

    """
            ID = str(sub['asset_id'])
            try: 
                #if prob >= threshold: status = 'detected'
                #else: status = 'rejected'
                #sample.member[ID].retiredby = 'machine'
                #sample.member[ID].state = 'inactive'
                #sample.member[ID].status = status
            except:
                noSWAP += 1


            # We can't do this with the current pickles... 
            # Initialize the subject in SWAP Collection
            ID = sub['asset_id']

            try: 
                test = sample.member[ID]
            except: 
                sample.member[ID] = swap.Subject(ID, str(sub['SDSS_id']),
                                                 category='test',
                                                 kind='test',
                                                 flavor='test',
                                                 truth='UNKNOWN',
                                                 thresholds=swap_thresholds, 
                                                 location=sub['external_ref'],
                                                 prior=prior) 

            # THIS NEEDS TO FUCKING CHANGE. =(
            if p >= threshold: 
                result = 'FEAT'
                status = 'detected'
            else: 
                result = 'NOT'
                status = 'rejected'

            sample.member[ID].was_described(by=MLbureau.member[Name], 
                                            as_being=result,
                                            at_time=time,
                                            while_ignoring=0,
                                            haste=True)
            
            # Try to jerry-rig something here....
            if p >= threshold: status = 'detected'
            else: status = 'rejected'

            try:
                sample.member[ID].retiredby = 'machine'
                sample.member[ID].state = 'inactive'
                sample.member[ID].status = status
            else:
                print "MC: subject {0} not found in collection. Bummer".format(ID)
            """

"""
labels, counts = np.unique(train_labels, return_counts=True)

majority = np.max(counts)

for label, count in zip(labels, counts):
if majority == count:
    major_idx = np.where(train_labels == label)[0]
    major_idx = major_idx[:np.sum(train_labels==1-label)]

    minor_idx = np.where(train_labels == 1-label)[0]

    train_features = np.concatenate([train_features[major_idx],
                                     train_features[minor_idx]])

    train_meta = np.concatenate([train_meta[major_idx],
                                 train_meta[minor_idx]])

    train_labels = np.concatenate([train_labels[major_idx],
                                   train_labels[minor_idx]])
"""