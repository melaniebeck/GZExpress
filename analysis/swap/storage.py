# ===========================================================================

import swap

import numpy as np
from astropy.table import Table
import machine_utils as ml
import pdb

# ======================================================================

class StorageLocker(object):
    """
    NAME
        StorageLocker

    PURPOSE
        Stores metadata associated with every subject in the sample.

        In the case of GZ, the intended purpose is to store various subject
           features necessary to train various Machine algorithms 
           (i.e., Concentration, Asymmetry, Gini, M20, ellipticity). 

        Additionally, this storage tracks flags which mark various subjects 
           as part of a specific subset (i.e., training sample, validation 
           sample, or test sample) denoted under the "MLsample" column as 
           'train', 'valid', or 'test'.

    COMMENTS
        All subjects in Storage are all Zooniverse subjects.

    INITIALISATION
        From scratch.

    METHODS
        StorageLocker.member(Name)     Returns the Subject called Name
        StorageLocker.size()           Returns the size of the Storage Locker
        StorageLocker.list()           Returns the IDs of the members

    BUGS

    AUTHORS
       Melanie Beck
       modelled after Collection.py by Phil Marshall. 
    """

# ----------------------------------------------------------------------------

    def __init__(self):

        metadata = Table.read('metadata_ground_truth_labels.fits')

        # set the validation sample
        metadata['MLsample'][metadata['Expert_label']!=-1] = 'valid'

        # set a priori training sample (testing purposes)
        #nair_not_expert = np.where( (metadata['Nair_label']!=-1) & 
        #(metadata['MLsample']=='test'))

        #metadata['MLsample'][nair_not_expert] = 'train'
        #print "ML: forceably setting a training sample for testing purposes"
        #print "ML: when finished remove this from storage.py"

        self.subjects = metadata

        return None

# ----------------------------------------------------------------------------

    def __str__(self):
        return 'storage locker containing %d subjects' % (self.size())

# ----------------------------------------------------------------------------
# Return the number of collection members:

    def size(self):
        return len(self.subjects)

# ----------------------------------------------------------------------------
# Return a complete list of collection members:

    def list(self):
        return self.subjects['asset_id']

# ----------------------------------------------------------------------------
# Return the requested subset (valid, train, test)

    def fetch_subsample(self, sample_type=None, class_label=None):

        if sample_type == 'test': 
            try:
                selection = (self.subjects['MLsample']==sample_type)
            except:
                selection = (self.subjects['MLsample']=='test ')
        else:
            selection = (self.subjects['MLsample']==sample_type)

        if np.sum(selection) > 0:
            subsample = self.subjects[selection]
        else:
            print "ML: no subsample of type %s found!"%sample_type
            return None 

        # If a class label is given, try to clean it up a bit -- 
        #   Not all subjects have labels for each possible label type
        try:
            subsample = subsample[subsample[class_label]!=-1]
        except: 
            print "ML: %s does not exist as a class label"%class_label
            print "ML: delivering the entire subsample"   

        return subsample 


    @staticmethod
    def extract_sample_features(self, data, keys=['M20', 'C', 'E', 'A', 'G'], 
                                whiten=False):
        '''
        INPUTS:
            astropy Table (or dictionary?)
        
        PURPOSE:
            This function isolates those rows which have well-measured
            morphological features that can be fed into the machine classifiers, 
            i.e. no nans or infs
    
        RETURNS:
            original astropy Table with only those entries that pass selection
            whitened np.array of the morphological parameters    
        '''
        morph = np.array([data[k] for k in keys], dtype='float32').T

        # remove those which don't have morph parameters measured
        training = ((~np.isnan(morph).any(1)) & (~np.isinf(morph).any(1)))

        if whiten:
            return data[training], whiten(morph[training])
        else:
            return data[training], morph[training]

       



