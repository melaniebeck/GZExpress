import swap
#import compare_SWAP_GZ2 as utils
from simulation import Simulation

import os, sys, subprocess, getopt
from argparse import ArgumentParser
from astropy.table import Table
import pdb
import datetime as dt
import numpy as np
import cPickle


def MachineShop(args):
    
    """
    Sometimes you just need to the run the Machine on a bunch of already
    made SWAP-runs / simulations. 
    If so, this script is for you!
    """
     
    # Get parameters from the SWAP run of interest
    the = swap.Configuration(args.config)
    params = the.parameters


    sim = Simulation(config=args.config, directory='.', variety='feat_or_not')

    # this was originally set to 2/17/09 which is WRONG
    first_day = dt.datetime(2009, 2, 12)
    today = dt.datetime.strptime(params['start'], '%Y-%m-%d_%H:%M:%S')
    start_day = dt.datetime(2009, 2, 17)
    last_day = dt.datetime.strptime(params['end'], '%Y-%m-%d_%H:%M:%S')
    yesterday = None


    run_machine = False
    SWAP_retired = 0
    notfound = 0
    last_night = None

    for idx, filename in enumerate(sim.retiredFileList[(today-first_day).days:]):
        print ""
        print "----------------------- The Machine Shop ----------------------------"
        print "Today is {}".format(today)

        if today >= last_day:
            print "Get outta the machine shop!"
            exit()

        # ---------------------------------------------------------------------
        #  OPEN METADATA PICKLE (updated each time MachineClassifier is run)
        # ---------------------------------------------------------------------

        backup_meta_file = params['metadatafile'].replace('.pickle', '_orig.pickle')

        if today == first_day:
            try:
                storage = swap.read_pickle(backup_meta_file,'metadata')
            except:
                print "MachineShop: Backup metadata pickle not yet created."
                print "MachineShop: Opening original metadata pickle file instead"
                storage = swap.read_pickle(params['metadatafile'],'metadata')

                if 'retired_date' not in storage.subjects.colnames:
                    storage.subjects['retired_date'] = '2016-09-10'

                if 'valid' not in np.unique(storage.subjects['MLsample']):
                    expert = (storage.subjects['Expert_label']!=-1)
                    storage.subjects['MLsample'][expert] = 'valid'

                # save an untouched copy for reference later
                print "MachineShop: Creating a backup metadata pickle"
                swap.write_pickle(storage, backup_meta_file)           
        else:
            storage = swap.read_pickle(params['metadatafile'],'metadata')

        # Regardless of which metadata you open, make sure it has these columns
        #       (old metadata files WON'T have them!)
        if 'retired_date' not in storage.subjects.colnames:
            storage.subjects['retired_date'] = '2016-09-10'

        if 'valid' not in np.unique(storage.subjects['MLsample']):
            expert = (storage.subjects['Expert_label']!=-1)
            storage.subjects['MLsample'][expert] = 'valid'

        subjects = storage.subjects

        # I just need to know what was retired TONIGHT --
        # compare what's retired UP TILL tonight with what was 
        # retired up till LAST NIGHT
        SWAP_retired_by_tonight = sim.fetchCatalog(filename)

        # If we're picking up where we left off, grab previous training sample
        #if today>start_day and last_night is None:
        #    print 'MachineShop: getting previous training sample'
        #    last_night = subjects[subjects['MLsample']=='train']
        #    last_night['zooid'] = last_night['SDSS_id']

        try:
            ids_retired_tonight = set(SWAP_retired_by_tonight['zooid']) - \
                                        set(last_night['zooid'])
        except:
            ids_retired_tonight = set(SWAP_retired_by_tonight['zooid'])


        print "Newly retired subjects: {}".format(len(ids_retired_tonight))

        # Now that I have the ids from the previous night, adjust the 
        # metadata file to reflect what was retired / add SWAP info
        for ID in list(ids_retired_tonight):

            # Locate this subject in the metadata file
            mask = subjects['SDSS_id'] == int(ID)

            # Update them in metadata file as training sample for MC
            # DOUBLE CHECK THAT IT HAS NOT BEEN RETIRED BY MACHINE!!!
            if subjects['MLsample'][mask] == 'test ':
                SWAP_retired+=1

                subjects['MLsample'][mask] = 'train'
                subjects['retired_date'][mask] = dt.datetime.strftime(today, '%Y-%m-%d')
                subjects['SWAP_prob'][mask] = SWAP_retired_by_tonight['P'][SWAP_retired_by_tonight['zooid']==ID]

                run_machine = True
            else:
                notfound +=1

        if len(subjects[subjects['MLsample']=='train'])>=50000:
            run_machine = True

        last_night = SWAP_retired_by_tonight

        print "Retired by this day:", len(last_night)

        print ""
        print "MachineShop: Found {0} subjects retired by SWAP on {1}"\
                .format(SWAP_retired, today)

        print "MachineShop: {0} total subjects retired so far"\
                .format(np.sum(subjects['MLsample']=='train'))

        print "MachineShop: Found {0} subjects retired by Machine."\
                .format(np.sum(subjects['MLsample']=='mclas'))

        print "MachineShop: Saving updated StorageLocker."
        
        # Save our new metadata file -- MC needs this -- save to NOT the original
        swap.write_pickle(storage, params['metadatafile'])


        if run_machine: 
            # Need to doctor the config to refect the "correct date" 
            params['start'] = today.strftime('%Y-%m-%d_%H:%M:%S')
            swap.write_config(args.config, params)
    
            # Run MachineClassifier.py using this subject file
            os.system("python MachineClassifier.py -c %s"%args.config)
            """os.system("python test_Machine.py -c {0}".format(args.config))"""
        

            # MachineClassifier updates the configfile so now we need to open the NEW one
            the = swap.Configuration(args.config)
            params = the.parameters

        # Update date (since we're not running SWAP)
        today += dt.timedelta(days=1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("-o", "--old", dest='old_run', action='store_true',
                        default=False)
    
    args = parser.parse_args()
    
    MachineShop(args)
    


    """ This whole approach is stupid.
    # Determine amount of SWAP output 
    # If there was a successful run, there should be 
    #   -- directories beginning with params['survey'] 
    #   -- pickle files beginning with params['trunk']
    
    detectfilelist = utils.fetch_filelist(params, kind='detected')
    
    if  args.old_run:
        rejectfilelist = utils.fetch_filelist(params, kind='retired')
    else:
        rejectfilelist = utils.fetch_filelist(params, kind='rejected')

    train_detected_ids = set()
    train_rejected_ids = set()
    
    for dfile, rfile in zip(detectfilelist, rejectfilelist):
        detected = utils.fetch_classifications(dfile)
        rejected = utils.fetch_classifications(rfile)
        
        detect_ids = set(detected['zooid'])
        reject_ids = set(rejected['zooid'])
        
        # Grab only the new ones
        new_detected = detect_ids.difference(train_detected_ids)
        print "%i new detected subjects"%len(new_detected)
        
        new_rejected = reject_ids.difference(train_rejected_ids)
        print "%i new rejected subjects"%len(new_rejected)
        
        # Loop through the new ids and switch from 'test' to 'train'
        for new in new_detected: 
            
            if subjects['MLsample'][subjects['SDSS_id']==new] != 'valid':
                subjects['MLsample'][subjects['SDSS_id']==new] = 'train'
                subjects['SWAP_prob'][subjects['SDSS_id']==new] = 1.
                
                
        for new in new_rejected:
                    
            if subjects['MLsample'][subjects['SDSS_id']==new] != 'valid':
                subjects['MLsample'][subjects['SDSS_id']==new] = 'train'
                subjects['SWAP_prob'][subjects['SDSS_id']==new] = 0.



        # ---------------------------------------------------------------------
        #  OPEN COLLECTION PICKLE (updated each time MachineClassifier is run)
        # ---------------------------------------------------------------------
        backup_col_file = params['samplefile'].replace('.pickle','_orig.pickle')
        if today == start_day:
            try: 
                # If we're starting fresh, we want to open the original file, if it exists
                collection = swap.read_pickle(backup_col_file,'collection')
            except:
                # If it doesn't exist, open up the regular file and ...
                print "MachineShop: backup collection pickle has not been made yet"
                print "MachineShop: opening original collection file"
                collection = swap.read_pickle(params['samplefile'],'collection')

                # Save the original collection file for comparison later
                if not os.path.isfile(backup_col_file):
                    print "MachineShop: creating a backup collection pickle"
                    swap.write_pickle(collection, backup_col_file)
        else:
            # If we're in the middle of the run, we want to open the file that's 
            # constantly being updated by MachineClassifier
            collection = swap.read_pickle(params['samplefile'],'collection')


        # ---------------------------------------------------------------------
        #  ISOLATE TRAINING SAMPLE -- SWAP-RETIRED BY "TODAY"
        # ---------------------------------------------------------------------
        for subjectID in collection.list():
            subject = collection.member[subjectID]

            if subject.retiredby == 'machine':
                machine_retired+=1

            if subject.retirement_time != 'not yet':
                date = dt.datetime.strptime(subject.retirement_time,'%Y-%m-%d_%H:%M:%S') 

                yesterday = today-dt.timedelta(days=1)

                if (date < today) and (date >= yesterday) and (subject.retiredby == 'swap'):

                    mask = subjects['SDSS_id'] == int(subject.ZooID)

                    # Update them in metadata file as training sample for MC
                    # DOUBLE CHECK THAT IT HAS NOT BEEN RETIRED BY MACHINE!!!
                    if subjects['MLsample'][mask] == 'test ':
                        SWAP_retired+=1

                        subjects['MLsample'][mask] = 'train'
                        subjects['retired_date'][mask] = dt.datetime.strftime(yesterday, '%Y-%m-%d')
                        subjects['SWAP_prob'][mask] = subject.mean_probability

                        run_machine = True

    """