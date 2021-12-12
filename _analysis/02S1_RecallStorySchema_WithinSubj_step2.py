from _functions import * #includes constants and score function
import os
import deepdish as dd

# from util import * #utility stuff
from _deconvolve import * #utility stuff

import brainiak.eventseg.event as event #HMM

import numpy as np #used a lot

import h5py #for saving final output

import sys 

import time


################
################ INPUTS
################

date = sys.argv[1] # 'date label to use'
hem = str(sys.argv[2]) # 'L' or 'R'
roi = sys.argv[3] # 'STS', 'SL', 'atlas'
roi_id = int(sys.argv[4]) # "9999" or int
scoretype = sys.argv[5] # "sl_percept_scores" or any directory name that makes sense
extra = sys.argv[6] #within_modality

# date = 20211014 #sys.argv[1] # 'date label to use'
# hem = 'R' #sys.argv[2] # 'L' or 'R'
# roi = 'SL' #sys.argv[3] # 'STS', 'SL', 'atlas'
# roi_id = 22 #int(sys.argv[4]) # "9999" or int
# scoretype = 'sl_recall_score' #sys.argv[5] # "sl_percept_scores" or any directory name that makes sense
# extra = 'within_modality'

################
################ CONSTANTS & DESIGN MATRIX
################

stories = ['Brazil',  'Derek', 'MrBean',   'PulpFiction',
           'BigBang', 'Santa', 'Shame',    'Vinny',
           'DueDate', 'GLC',   'KandD',    'Nonstop',
           'Friends', 'HIMYM', 'Seinfeld', 'UpInTheAir']

nSubj = 30 # lost recall data to subject 31 due to signal loss
TR = 1.5
nStories = len(stories)
nEvents = 4
nPerm = 1000
nv = 40962

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
################ FUNCTIONS 
################



################
################ RUN ANALYSIS
################

start_time = time.time()

## load ROI verts
roi_verts = ExtractROI(roi, roi_id, hem)

###
####### GET GENERAL VALID VERTS (20210913
###

# braindata_path = '/jukebox/norman/rmasis/clones/SchemaBigFiles/FullPerceptRecall'
braindata_path = '../../SchemaBigFiles/FullPerceptRecall'


if 'hippo' not in roi:
    percept = dd.io.load(os.path.join(braindata_path,'wholebrain_perception.h5'),'/'+hem) 
else:
    percept = dd.io.load(os.path.join(braindata_path,"hipp_perception.h5"))
      
# percept = dd.io.load(os.path.join(braindata_path,'wholebrain_perception.h5'),'/'+hem) 

gen_valid_verts = roi_verts.copy()
for story in stories:
    gen_valid_verts = gen_valid_verts * ~np.any(np.any(np.isnan(percept[story]), axis=2), axis=1)
del(percept) #don't need it after getting valid verts
    
    
## load recall timeseries
if 'hipp' in roi:
    recall = dd.io.load(os.path.join(braindata_path,"hipp_perception_recall.h5"))['recall']
else:
    recall = dd.io.load(os.path.join(braindata_path,'wholebrain_recall.h5'),'/'+hem)

### load perception templates (for within subjects HMM fitting)
percept_dirname = '{}_percept_score'.format(scoretype.split('_')[0].lower())
# path = '../SchemaBigFiles/draft_PAPER/{}'.format(percept_dirname)
path = '../../SchemaBigFiles/_PaperOutputData/{}'.format(percept_dirname)

fname = '{date}_{roi}_{roi_id:04d}_{hem}_{percept_dirname}_{extra}.h5'.format(date=date, roi=roi, roi_id=roi_id,
                                                            hem=hem, percept_dirname=percept_dirname,extra=extra)
fullpath = os.path.join(path, fname)
# story_templates = dd.io.load(fullpath, '/' + hem + '/' + 'group_event_templates')
story_templates = dd.io.load(fullpath, '/' + hem + '/' + 'subj_event_templates')[has_R,:,:,:] #for individual subject analysis, i think.
#for individual subject analysis, i think.


### RUN HMM FITTING PROCEDURE
dim = roi_verts.sum()
print(dim)


template_match = np.zeros((nStories, nStories, 4, nSubj))
hmm_overall_match = np.zeros((nStories,nStories,nSubj))

## iterate through every subject
for s,subj in enumerate((subjects)):
    
    ## iterate through story to use as template
    for st, story in enumerate(stories):
        
#         template = story_templates[st,:,:] #template from perception
        template = story_templates[s,st,:,:] #for individual subjects, dimension mismatch
        
        nTRs = recall[story][str(s)].shape[1] #total TRs in timeseries of this story's recall
        
        ## declare HMM object
        ev = event.EventSegment(nEvents) #each story has 4 schematic subevents

        ## grab perception timeseries of current subject to calculate variance parameter 
        ## (there could be nan vertices for a particular story for a particular subject for specified ROI)
#         percept = dd.io.load(os.path.join(braindata_path,'wholebrain_perception.h5'),'/'+hem+'/'+ story, 
#                       dd.aslice[np.where(roi_verts)[0],:,s], unpack=True)

        ## grab perception timeseries of current subject to calculate variance parameter 
        ## (there could be nan vertices for a particular story for a particular subject for specified ROI)
        if 'hippo' not in roi:
#             percept = dd.io.load(os.path.join(braindata_path,'wholebrain_perception.h5'),'/'+hem+'/'+ story, 
#                           dd.aslice[np.where(gen_valid_verts)[0],:,:], unpack=True)[:,:,has_R][:,:,s]
            percept = dd.io.load(os.path.join(braindata_path,'wholebrain_perception.h5'),'/'+hem+'/'+ story, 
                          dd.aslice[np.where(roi_verts)[0],:,:], unpack=True)[:,:,has_R][:,:,s]
        else:
#             percept = dd.io.load(os.path.join(braindata_path,"hipp_perception.h5"), '/' + story,
#                                  dd.aslice[np.where(gen_valid_verts)[0],:,:], unpack=True)[:,:,has_R][:,:,s]
            percept = dd.io.load(os.path.join(braindata_path,"hipp_perception.h5"), '/' + story,
                                 dd.aslice[np.where(roi_verts)[0],:,:], unpack=True)[:,:,has_R][:,:,s]
        
        print(template.shape, percept.shape)

        ### !! COMBACK AND REVISIT; less precise b/c we haven't found out whether recall timeseries has same valid verts
        valid_verts = ~np.isnan(percept[:,0]) * ~np.isnan(template[:,0])
        ev_var1 = ev.calc_weighted_event_var(percept[valid_verts].T, design_intact[st][:,1:], template[valid_verts,:])
        print("VALID VERTS 1: ", valid_verts.sum())
#         print(subj, np.any(np.isnan(percept)),np.any(np.isnan(template)),np.all(np.isnan(percept)),np.all(np.isnan(template)), ev_var1)

        ## iterate through story to fit with template
        for st_j, story_j in enumerate(stories):

            recall_timeseries = recall[story_j][str(s)][roi_verts] 
#             recall_timeseries = recall[story_j][str(s)][gen_valid_verts] 
            
            valid_verts = ~np.isnan(percept[:,0]) * ~np.isnan(template[:,0]) * ~np.isnan(recall_timeseries[:,0])
            print("VALID VERTS 2: ", valid_verts.sum())

            ev.set_event_patterns(template[valid_verts]) #provide HMM with story template of shape: (voxel,event)

            ### !!!!!!!!!!!
            ### !! COME BACK REVISIT; more precise b/cj we are using non nans in both percept and recall timeseries
            ev_var = ev.calc_weighted_event_var(percept[valid_verts].T, design_intact[st][:,1:], template[valid_verts])
            print('... ', ev_var1==ev_var)
            print(ev_var)
            print(recall_timeseries[valid_verts].T)
            
            
            ev_fit = ev.find_events(recall_timeseries[valid_verts].T, var=ev_var)[0] ## this fixes the problem of not finding events per subject!

            #plt.figure()
            #plt.plot(ev_fit); plt.title(stories[story_j])
            #labels = np.argmax(ev_fit, axis=1) 
            #bounds = np.where(np.diff(labels))[0]
            #bounds_aug = np.concatenate(([0],bounds,[nTRs]))

            weighted_vox = np.matmul(recall_timeseries[valid_verts],ev_fit) #(voxel x time) * (time, event) --> (voxel x event)

            ###
            ### SCORING FOR HMM IN HOW WELL IT DID AT MATCHING PERCEPTION TO RECALL FOR THIS STORY
            ###

            #events.T = (4,nVox), weighted_vox.T = (nVox, 4)
            cc = np.corrcoef(template[valid_verts].T, weighted_vox.T)[:nEvents, nEvents:] # the .T gives us both --> (nVox,events) and (events,vox)
            print(np.isnan(weighted_vox.ravel()).sum(), np.isnan(template[valid_verts].ravel()).sum(),
                 np.isnan(cc.ravel()).sum())
            ### SYMMETRIC template match score
            for i in range(nEvents):
                template_match[st,st_j,i,s] = cc[i,i] - np.nanmean(cc[i,np.arange(nEvents)!=i],axis=0)/2 - np.nanmean(cc[np.arange(nEvents)!=i,i],axis=0)/2 

            print('TEMPLATE match: ', np.isnan(template_match[st,st_j,i,s]))
            ### OLD WAY TO CALCULATE template match
            d = np.diag(cc).sum()
            calc = d/nEvents - (cc.sum()-d)/(nEvents**2 - nEvents) #diagonal_mean minus off-diagonal_mean
            hmm_overall_match[st,st_j,s] = calc #the score of this hmm at trying to match perception to recall


            #vartup =tuple(['{:.3f}'.format(ev_var[i]) for i in range(len(ev_var))])
            #plt.figure();plt.imshow(cc);plt.colorbar();plt.title('score: {:.3f},subj: {},st: {}\n var:{}'.format(calc, subj,stories[story],vartup))
            #plt.figure();plt.imshow(cc);plt.colorbar();plt.title('score: {:.3f},subj: {},st: {}\n'.format(calc, subj,stories[story]))
            #plt.show()
            
            
            
## get story and schema scores for every event
story_effect = np.zeros((nSubj,nStories,nEvents,nPerm+1)) 
schema_effect = np.zeros((nSubj,nStories,nEvents,nPerm+1)) 

for ev in range(nEvents):
    story_effect[:,:,ev,:], schema_effect[:,:,ev,:] = GetStorySchemaEffect(template_match[:,:,ev,:])
    

################
################ SAVE DATA
################

# path = '../SchemaBigFiles/draft_PAPER'
path = '../../SchemaBigFiles/_PaperOutputData'


# date = 20210930 # within subj

path = os.path.join(path,scoretype)
if not os.path.exists(path):
    print("...making directory: ", path)
    os.makedirs(path)    
else:
    print("...directory exists: ", path)

fname = '{date}_{roi}_{roi_id:04d}_{hem}_{scoretype}_{extra}_within_subj.h5'.format(date=date, roi=roi, roi_id=roi_id,
                                                            hem=hem, scoretype=scoretype,extra=extra)
fullpath = os.path.join(path, fname)

print('...Saving to ', fullpath)

data_labels = ['reinstatement_similarity_matrix',
              'story_effect',
              'schema_effect']

reinstatement_measures = [template_match, # nStories, nStories, 4, nSubj
                         story_effect, # (30,16,4,1001)
                         schema_effect] # (30,16,4,1001)

with h5py.File(fullpath, 'w') as hf:

    # create dict entry
    hf = hf.create_group(hem)
    
    # create sub-dict entries
    for label,measure in zip(data_labels,reinstatement_measures): 
        hf.create_dataset(label, data=measure)
        
print('...SAVING COMPLETE.')


print((time.time() - start_time)/60)

