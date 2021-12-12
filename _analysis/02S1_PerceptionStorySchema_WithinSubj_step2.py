
import sys
# sys.path.append('../')
# #import time

import numpy as np
import deepdish as dd
import os
from scipy.stats import zscore

import nibabel as nib #use to load mmp atlas
# from itertools import compress #dont think i use this. probably not necessary
import h5py

from _deconvolve import *

from _functions import * #import the story schema effect scoring function



################
################ INPUTS
################

date = sys.argv[1] # 'date label to use'
hem = str(sys.argv[2]) # 'L' or 'R'
roi = sys.argv[3] # 'STS', 'SL', 'atlas'
roi_id = int(sys.argv[4]) # "9999" or int
scoretype = sys.argv[5] # "sl_percept_scores" or any directory name that makes sense
extra = sys.argv[6] #within_modality


################
################ CONSTANTS
################

nSubj = 31 # perception has 31 subj
TR = 1.5
nStories = len(stories)
nEvents = 4
nPerm = 1000
nv = 40962

################
################ RUN ANALYSIS
################

### load perception templates
percept_dirname = '{}_percept_score'.format(scoretype.split('_')[0].lower())
# path = '../SchemaBigFiles/draft_PAPER/{}'.format(percept_dirname)
path = '../../SchemaBigFiles/_PaperOutputData/{}'.format(percept_dirname)

fname = '{date}_{roi}_{roi_id:04d}_{hem}_{percept_dirname}_{extra}.h5'.format(date=date, roi=roi, roi_id=roi_id,
                                                            hem=hem, percept_dirname=percept_dirname,extra=extra)
fullpath = os.path.join(path, fname)
# story_templates = dd.io.load(fullpath, '/' + hem + '/' + 'group_event_templates')
story_templates = dd.io.load(fullpath, '/' + hem + '/' + 'subj_event_templates') #for individual subject analysis, i think.



# ### RUN SIMILARITY MATRIX (within subj for schema score)
# within_subj_smat = np.zeros((nStories,nStories,nEvents,nSubj))
# for s in range(nSubj):
#     story_ev_patterns = story_templates[s,:,:,:]       
#     for ev in range(nEvents):
#         within_subj_smat[:,:,ev,s] = np.corrcoef(story_ev_patterns[:,:,ev], story_ev_patterns[:,:,ev])[:16,16:]
        
    
### RUN SIMILARITY MATRIX (within subj for schema score) 2021, nov 22
within_subj_smat = np.zeros((nStories,nStories,nEvents,nSubj))
for s in range(nSubj):
    story_ev_patterns = story_templates[s,:,:,:] 
    
    for ev in range(nEvents):
        ### make sure that if a particular subj has less valid verts, that we only create correlation matrix where valids exist
        valid_verts = np.all(~np.isnan(story_templates[s,:,:,:]),0)[:,ev] #nov 22, 2021
        within_subj_smat[:,:,ev,s] = np.corrcoef(story_ev_patterns[:,valid_verts,ev], story_ev_patterns[:,valid_verts,ev])[:16,16:]
        
        
### RUN STORY AND SCHEMA SCORING (although STORY score pointless here)
nEvents= 4
story_effect = np.zeros((nSubj,nStories,nEvents,nPerm+1)) 
schema_effect = np.zeros((nSubj,nStories,nEvents,nPerm+1)) 

for ev in range(nEvents):
    story_effect[:,:,ev,:], schema_effect[:,:,ev,:] = GetStorySchemaEffect(within_subj_smat[:,:,ev,:])
    
################
################ FILE SAVING 
################

# path = '/jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER'
path = '../../SchemaBigFiles/_PaperOutputData'


# if roi in ['mPFC', 'PMC', 'Aud', 'MTN', 'PM', 'MP','AT','STS','SFG']:
#     savedir = 'roi_{}'.format(scoretype)
# elif roi == 'atlas':
#     savedir = 'roi_{}'.format(scoretype)
# elif roi.lower() == 'sl':
#     savedir = 'sl_{}'.format(scoretype)

path = os.path.join(path,scoretype)
if not os.path.exists(path):
    print("...making directory: ", path)
    os.makedirs(path)    
else:
    print("...directory exists: ", path)
    
    
data_labels = [
               'loo_event_similarity_matrix', 
               'story_effect',
               'schema_effect']

perception_measures = [
                       
                       within_subj_smat,  # (16,16,4,31) #N-1 spatial ISC 
                       story_effect, 
                       schema_effect]

# date = 20210930 #for within subj

fname = '{date}_{roi}_{roi_id:04d}_{hem}_{scoretype}_{extra}_within_subj.h5'.format(date=date, roi=roi, roi_id=roi_id,
                                                            hem=hem, scoretype=scoretype,extra=extra)
fullpath = os.path.join(path, fname)
print('...Saving to ', fullpath)

with h5py.File(fullpath, 'w') as hf:

    # create dict entry
    hf = hf.create_group(hem)
    
    # create sub-dict entries
    for label,measure in zip(data_labels,perception_measures): 
        hf.create_dataset(label, data=measure)
        
print('...SAVING COMPLETE.')

            