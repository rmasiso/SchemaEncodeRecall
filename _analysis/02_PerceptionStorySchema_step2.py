####
######## GROUP templates are in ROI verts and not in valid verts of percept.. is this something i need to change? 
######## realized this when the subj temlates had like 274 vox and group had 324 for SL 22 right hem (20210913)
######## ok so its ironic that for the subj templates i have fewer vertices
######## this is because i do the valid verts check for all subj and then use that on the perception stuff of each subj
######## when i should do the opposite, have the group have this thing.. need to fix this..?
######## make sure that it wasnt on purpose.. but in anycase, i do use valid verts for group, i just made total array big via roi_verts rather than valid_verts


import sys
# sys.path.append('../')
# #import time

import numpy as np
import deepdish as dd
import os
from scipy.stats import zscore

import nibabel as nib #use to load mmp atlas

import h5py

from _deconvolve import *

from _functions import * #import the story schema effect scoring function

################
################ INPUTS
################


# date = 20211014
# hem = 'R'
# roi = 'SL'
# roi_id = 22
# scoretype = 'sl_percept_scores'
# modality_mode = 'within_modality'

date = sys.argv[1] # 'date label to use'
hem = str(sys.argv[2]) # 'L' or 'R'
roi = sys.argv[3] # 'STS', 'SL', 'atlas'
roi_id = int(sys.argv[4]) # "9999" or int
scoretype = sys.argv[5] # "sl_percept_scores" or any directory name that makes sense
modality_mode = str(sys.argv[6]) #within_modality or across_modality


################
################ FUNCTIONS 
################

def RunPerceptionSimilarityMatrix(roi_verts,percept,hem):
    '''
    roi_verts is False bool array of shape 40962 with TRUE indicating ROI vertices
    timeseries is dict of hem and story for all subjects
    '''
    

    valid_verts = roi_verts.copy()
    for story in stories:
        valid_verts = valid_verts * ~np.any(np.any(np.isnan(percept[story]), axis=2), axis=1) #(np.any(np.all(~np.isnan(percept[story]), axis=2), axis=1))#(~np.any(np.all(np.isnan(percept[story]), axis=2), axis=1))#(np.any(np.all(~np.isnan(percept[story]), axis=2), axis=1))

    print(roi_verts.sum(), valid_verts.sum())

    dim = valid_verts.sum()
    
    dim = roi_verts.sum() #coommented nov 22 2021 (issue with L ang subj 20 (or 19 in reduced)
    
    subj_templates = np.zeros((nSubj,nStories, dim, 4)) # hold betas for each subject // 20201127

    ### set up event template array and run GLM for full N group (will serve as template later)
    group = np.full((nStories, roi_verts.sum(), 4), fill_value=np.nan) #group average of betas (includes left out subject)
    for st,story in enumerate(stories):
        group_valid = np.logical_and(roi_verts,np.all(~np.isnan(percept[story][:,0,:]),axis=1)) 
        group_idx = np.arange(valid_verts.sum()) #useful for indexing
#         group_mean = np.nanmean(percept[story][roi_verts], axis=2)
        group_mean = np.nanmean(percept[story][valid_verts], axis=2)
        group[st, group_idx, :] = zscore(deconv(group_mean, design_intact[st])[:, 1:], axis=0, ddof=1)

    ### set up similarity matrix array
    loo_story_corr = np.zeros((nStories,nStories,nSubj)) #dont really use this. # REMOVE later?
    loo_ev_corr = np.zeros((nStories, nStories,4,nSubj))

    ### Create similarity matrix for each left-out subject
    for s in (range(nSubj)):

        ### LOO group stuff
        loo_subj_betas = np.full((nStories, dim, 4),fill_value=np.nan)
        loo_group_betas = np.full((nStories, dim, 4), fill_value=np.nan)

        ### estimate betas for left-out-subject's stories
        for st,story in enumerate(stories):
            
            ### gather valid_verts across the participants for this story
            valid_verts = np.logical_and(roi_verts,(~np.isnan(percept[story][:,0,s]))) 
            valid_idx = np.arange(valid_verts.sum())

            ### get left-out-subject's story timeseries at perception
            loo_subj = percept[story][valid_verts][:,:,s]  

            ### get average timeseries for this story across N-1 group
            loo_group_mean = np.nanmean(percept[story][valid_verts][:,:,np.arange(nSubj)!=s], axis=2)

            ### Run GLM for loo_subj and loo_group; maximize valid verts for later within-subj templates
            loo_subj_betas[st,valid_idx,:] = zscore(deconv(loo_subj, design_intact[st])[:, 1:], axis=0, ddof=1)
            loo_group_betas[st,valid_idx,:] =  zscore(deconv(loo_group_mean[:,:], design_intact[st])[:, 1:], axis=0, ddof=1)
            
            subj_templates[s,st,:,:] = loo_subj_betas[st,:,:]

        ### calculate similarity matrix
        #### run correlation of left-out-subject's stories with N-1 group's stories
        for ev in range(4):
                                         
            ### Check for lowest common denominator of valid vertices across stories
            valids_all = np.all(~np.isnan(loo_subj_betas[:,:,ev]),axis=0)                                         
                                       
            print('valids_all: ',valids_all.sum())

            ### Correlate every event of loo_subj with N-1 group
            loo_ev_corr[:,:,ev,s] = np.dot(loo_subj_betas[:,valids_all,ev], loo_group_betas[:,valids_all,ev].T)/(dim-1)#(dim-1)

            ### symmetrize
            loo_ev_corr[:,:,ev,s] = (loo_ev_corr[:,:,ev,s] + loo_ev_corr[:,:,ev,s].T)/2 #make N-1 mat symmetric per event

    return group, subj_templates, loo_ev_corr



# def RunPerceptionSimilarityMatrix(roi_verts,timeseries,hem):
#     '''
#     roi_verts is False bool array of shape 40962 with TRUE indicating ROI vertices
#     timeseries is dict of hem and story for all subjects
#     '''

#     dim = roi_verts.sum() # identify number of vertices in specified ROI

#     ### set up event template array
#     group = np.zeros((nStories, dim, 4)) #group average including left out subject 
#     for st, story in enumerate(stories):
#         ### Run GLM for full N group (will serve as template later)
#         group_mean = (np.nanmean(percept[story][roi_verts][:,:,:], axis=2))
#         group[st, :, :] = zscore(deconv(group_mean, design_intact[st])[:, 1:], axis=0, ddof=1)


#     ### set up similarity matrix array
#     loo_story_corr = np.zeros((nStories,nStories,nSubj)) #dont really use this. # REMOVE later?
#     loo_ev_corr = np.zeros((nStories, nStories,4,nSubj))

#     #     group_ev_corr = np.zeros((nStories, nStories,4,nSubj))

#     for s in tqdm_notebook(range(nSubj)):

#         ### LOO group stuff
#         loo_subj_betas = np.full((nStories, dim, 4),fill_value=np.nan)
#         loo_group_betas = np.full((nStories, dim, 4), fill_value=np.nan)

#         ### ALL SUBJECTS GROUP STUFF
#         group_betas = np.zeros((nStories, dim, 4))

#         ### estimate betas for left-out-subject's stories
#         for st,story in enumerate(stories):
#             # find valid verts for this subject that intersect with our region of interest
#             valid_verts = np.logical_and(roi_verts,(~np.isnan(percept[story][:,0,s]))) 
#             valid_idx = np.arange(valid_verts.sum()) #useful for indexing

#             ### get left-out-subject's story timeseries at perception
#             loo_subj = percept[story][valid_verts][:,:,s]  

#             ### get average timeseries for this story across N-1 group
#             loo_group_mean = np.nanmean(percept[story][valid_verts][:,:,np.arange(nSubj)!=s], axis=2)

#             ### Run GLM for loo_subj and loo_group 
#             loo_subj_betas[st,valid_idx,:] = zscore(deconv(loo_subj, design_intact[st])[:, 1:], axis=0, ddof=1)
#             loo_group_betas[st,valid_idx,:] =  zscore(deconv(loo_group_mean[:,:], design_intact[st])[:, 1:], axis=0, ddof=1)


#         ### calculate similarity matrix
#         #### run correlation of left-out-subject's stories with N-1 group's stories
#         print()
#         for ev in range(4):

#             ### Check for lowest common denominator of valid vertices across stories
#             allstoryvalids = np.all(~np.isnan(loo_subj_betas[:,:,ev]),axis=0)
#             print(allstoryvalids.sum())

#     #         allstoryvalids = [(~np.isnan(loo_subj_betas[st,:,ev])) for st in range(16)]
#     #         allstoryvalids = np.where(np.all(np.stack(allstoryvalids,axis=0),axis=0))[0]
#     # #         allstoryvalids = np.arange(np.all(~np.isnan(loo_subj_betas[:,:,ev]),axis=0).sum())
#     #         print(np.any(~np.isnan(loo_subj_betas[:,:,ev]),axis=0).sum(),np.all(~np.isnan(loo_subj_betas[:,:,ev]),axis=0).sum())
#     #         print(s, len(np.unique(allstoryvalids)), allstoryvalids.shape)
#     #         print((valid_idx==allstoryvalids).sum())

#             ### Correlate every event of loo_subj with N-1 group
#             loo_ev_corr[:,:,ev,s] = np.dot(loo_subj_betas[:,allstoryvalids,ev], loo_group_betas[:,allstoryvalids,ev].T)/(len(allstoryvalids)-1)#(dim-1)

#             ### symmetrize
#             loo_ev_corr[:,:,ev,s] = (loo_ev_corr[:,:,ev,s] + loo_ev_corr[:,:,ev,s].T)/2 #make N-1 ISC symmetric per event

#     return group, loo_ev_corr

################
################ CONSTANTS & DESIGN MATRIX
################

stories = ['Brazil',  'Derek', 'MrBean',   'PulpFiction',
           'BigBang', 'Santa', 'Shame',    'Vinny',
           'DueDate', 'GLC',   'KandD',    'Nonstop',
           'Friends', 'HIMYM', 'Seinfeld', 'UpInTheAir']

nSubj = 31
TR = 1.5
nStories = len(stories)

# Story event boundaries (in seconds)
events_seconds_intact = np.array([
     [0,     6,    46,    73,   115,   184],
     [0,     6,    30,   111,   169,   188],
     [0,     6,    21,    74,   159,   186],
     [0,     6,    40,    56,   171,   185],
     [0,     6,    31,    46,   156,   171],
     [0,     6,    71,   104,   165,   175],
     [0,     6,    26,    94,   134,   175],
     [0,     6,    28,    59,   151,   179],
     [0,     6,    51,   111,   122,   189],
     [0,     6,    23,    70,    84,   171],
     [0,     6,    53,    82,   136,   184],
     [0,     6,    29,    77,   123,   190],
     [0,     6,    30,    97,   138,   163],
     [0,     6,    36,   109,   124,   169],
     [0,     6,    39,    85,   107,   173],
     [0,     6,    33,   108,   144,   187]])

# Story lengths (in TRs)
story_TRs = np.array([124, 126, 125, 124,
                      115, 118, 118, 120,
                      127, 115, 124, 128,
                      110, 114, 116, 126])


# Compute design matrices for all stories
design_intact = []
for i in range(len(stories)):
    subevent_E = np.zeros((events_seconds_intact[i, 5], 5))
    for e in range(5):
        subevent_E[events_seconds_intact[i, e]:
                   (events_seconds_intact[i, e + 1] + 1), e] = 1
    design_intact.append(regressor_to_TR(subevent_E[:,:], TR, story_TRs[i]))
design_intact = np.array(design_intact)


################
################ RUN ANALYSIS
################

# ### IMPORT perception timeseries
# percept_path = '/jukebox/norman/rmasis/clones/SchemaBigFiles/FullPerceptRecall'
# percept = dd.io.load(os.path.join(percept_path,'wholebrain_perception.h5'),'/'+hem) #perception timeseries, shape for a story: (nTRs,40962)

percept_path = '../../SchemaBigFiles/FullPerceptRecall'

### IMPORT perception timeseries
if 'hippo' in roi:
#     percept = dd.io.load(os.path.join(percept_path,"hipp_perception_recall.h5"))['perception']
    percept = dd.io.load(os.path.join(percept_path,"hipp_perception.h5"))
else:
    percept = dd.io.load(os.path.join(percept_path,'wholebrain_perception.h5'),'/'+hem) #perception timeseries, shape for a story: (nTRs,40962)


### GET VERTS FOR THIS ROI (whether searchlight, atlas or specific roi)
roi_verts = ExtractROI(roi,roi_id,hem)


### RUN SIMILARITY MATRIX
group, subj_templates, loo_ev_corr = RunPerceptionSimilarityMatrix(roi_verts,percept,hem)

### RUN STORY AND SCHEMA SCORING
nEvents= 4
story_effect = np.zeros((nSubj,nStories,nEvents,nPerm+1)) 
schema_effect = np.zeros((nSubj,nStories,nEvents,nPerm+1)) 

### get story and schema scores for each event
for ev in range(nEvents):
#     story_effect[:,:,ev,:], schema_effect[:,:,ev,:] = GetStorySchemaEffect(loo_ev_corr[:,:,ev,:],test_mod='across_modality')
    story_effect[:,:,ev,:], schema_effect[:,:,ev,:] = GetStorySchemaEffect(loo_ev_corr[:,:,ev,:],test_mod=modality_mode)
    
################
################ FILE SAVING 
################

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
    
    
data_labels = ['group_event_templates',
               'subj_event_templates',
               'loo_event_similarity_matrix', 
               'story_effect',
               'schema_effect']

perception_measures = [group, # (16,nVerts,4)
                       subj_templates, #(31,16,nVerts,4)
                       loo_ev_corr,  # (16,16,4,31) #N-1 spatial ISC 
                       story_effect, 
                       schema_effect]

fname = '{date}_{roi}_{roi_id:04d}_{hem}_{scoretype}_{modality_mode}.h5'.format(date=date, roi=roi, roi_id=roi_id,
                                                            hem=hem, scoretype=scoretype,modality_mode=modality_mode)
fullpath = os.path.join(path, fname)
print('...Saving to ', fullpath)

with h5py.File(fullpath, 'w') as hf:

    # create dict entry
    hf = hf.create_group(hem)
    
    # create sub-dict entries
    for label,measure in zip(data_labels,perception_measures): 
        hf.create_dataset(label, data=measure)
        
print('...SAVING COMPLETE.')



