import sys
import numpy as np
import deepdish as dd
import os
from scipy import stats
from scipy.stats import zscore,pearsonr
from _functions import * #includes constants and score function


################
################ FUNCTIONS
################

def GetStorySchemaScore(date,roi,roi_id,scoretype,eff,event='all',hems='all'):

    path = '../../SchemaBigFiles/_PaperOutputData/{}'.format(scoretype)
    
    extra = 'within_modality'

    subj_selection = has_R if 'percept' in scoretype.split('_') else np.full(30,fill_value=True) #recalls only have 30 subj


    score = {}
   
    for hem in ['L','R']:
        fname = '{date}_{roi}_{roi_id:04d}_{hem}_{scoretype}_{extra}.h5'.format(date=date, roi=roi, roi_id=roi_id,
                                                                hem=hem, scoretype=scoretype,extra=extra)
        
        score[hem] = dd.io.load(os.path.join(path,fname), '/' + hem + '/'+ eff )[subj_selection,:]

        if event=='all':
            score[hem] = np.nanmean(score[hem][:,:,:,0],axis=2) #average out the event dimension, select raw data (not permtuation)
        else:
            score[hem] = score[hem][:,:,event,0]

    if hems =='all':
        seed = np.nanmean(np.concatenate((score['L'][np.newaxis,:],score['R'][np.newaxis,:])),axis=0) #average out left and right hemisphere seed
    else: #if hems == 'L' or'R'
        seed = score[hems]
        
    return seed


# def permute(x,y):
#     ####
#     #### PERMUTATION
#     ####

#     nPerm = 1000
#     corrs= np.zeros((nPerm+1))
#     betas = np.zeros((nPerm+1))

#     regr = linear_model.LinearRegression(fit_intercept=True)
#     regr.fit(x[:,np.newaxis], y[:,np.newaxis])
#     betas[0] = regr.coef_[0][0]
#     corrs[0] = pearsonr(x,y)[0]
    
#     for p in range(nPerm):
#         perm_idx = np.random.permutation(len(y))
#         corrs[p+1] = pearsonr(x,y[perm_idx])[0]

#         # Create linear regression object
#         regr = linear_model.LinearRegression(fit_intercept=True)
#         regr.fit(x[:,np.newaxis], y[perm_idx,np.newaxis])
#         betas[p+1] = regr.coef_[0][0]


#     psig_corr = np.sum(corrs>corrs[0])/(nPerm+1)
#     psig_beta = np.sum(betas>betas[0])/(nPerm+1)
    
#     return corrs,betas,psig_corr,psig_beta


def permute(x,y):
    ####
    #### PERMUTATION
    ####
    
    
#     print(x.shape,y.shape)
    nPerm = 1000
    corrs= np.zeros((nPerm+1))
    betas = np.zeros((nPerm+1))

    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x[:,np.newaxis], y[:,np.newaxis])
    betas[0] = regr.coef_[0][0]
    corrs[0] = pearsonr(x,y)[0]
    
    for p in range(nPerm):
        ## shuffle by fully breaking y
#         perm_idx = np.random.permutation(len(y))

        ## shuffle by stories within subject
        perm_idx = np.array([np.random.permutation(np.arange(16)+(16*i)) for i in range(30)]).ravel()
        
#         print(x.shape,y.shape,y[perm_idx].shape)
        corrs[p+1] = pearsonr(x,y[perm_idx])[0]

        # Create linear regression object
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(x[:,np.newaxis], y[perm_idx,np.newaxis])
        betas[p+1] = regr.coef_[0][0]


    psig_corr = np.sum(corrs>corrs[0])/(nPerm+1)
    psig_beta = np.sum(betas>betas[0])/(nPerm+1)
    
    return corrs,betas,psig_corr,psig_beta

################
################ INPUTS
################

date = sys.argv[1] #20200626

seed_roi = sys.argv[2] #'mPFC'
seed_id = 9999 #seeds will always be ROIs, so seed id will be 9999
seed_scoretype = sys.argv[3] #'roi_recall_score' #the seed
seed_eff = sys.argv[4] #'story_effect'

post_roi = sys.argv[5] #'SL' # could also be atlas, 
post_scoretype = sys.argv[6] #'sl_recall_score'
post_eff = sys.argv[7] #'schema_effect'

nSubj = 30

subset = [] #subset choice


################
################ ANALYSIS
################


ROIlist, nROIs = GetROIlist(post_roi) # choose ROIlist based on analysis type

### RUN SEARCHLIGHT

### include all subjects if no subset provided.
subset = subset if len(subset)!=0 else np.ones((nSubj,nStories),dtype=bool)

### array to hold the correlation values for every permutation for every searchlight
corr_p = {'L':np.zeros((len(ROIlist['L']),nPerm+1)),'R':np.zeros((len(ROIlist['R']),nPerm+1))}

### the seed ROI. we will correlate every searchlight with this seed
seed = GetStorySchemaScore(date,seed_roi,seed_id,seed_scoretype,seed_eff)

for hem in ['L','R']:
    for post_id in (range(len(ROIlist[hem]))): #run through every ROI in the dict of lists 

        post = GetStorySchemaScore(date, post_roi, post_id, post_scoretype, post_eff)

#         x = seed[subset].ravel()
#         y = post[subset].ravel()
        
        ## FLIPPING THIS for it to make sense...
        y = seed[subset].ravel()
        x = post[subset].ravel()

        corr,beta,psigcorr,psigbeta = permute(x,y) #run correlation
        corr_p[hem][post_id] = corr #save correlation value


################
################ FDR and SAVE
################

nv = 40962 #verts in fsaverage6 brain
raw_vox = {} # raw values
p_vox = {} # non-parametric p-values
q_vox = {} # q-values from FDR-correction

dd_vox = SLtoVox(corr_p, ROIlist, nv, zeronan=False)

for hem in ['L','R']:
    p_vox[hem] = NonparametricP(dd_vox[hem],sided=2)
    raw_vox[hem] = dd_vox[hem][:,0]
    
q_vox = FDR_p_hem(p_vox)

print('complete.')

###
### SAVE  SAVE SAVE
###

# date = 20200816 #two-sided seeds 
    
data = {'raw_vox':raw_vox, 'p_vox':p_vox, 'q_vox':q_vox} #package the results up

path = '../../SchemaBigFiles/_PaperOutputData/Brainmaps'
fname ='{}_{}_{}_{}_to_{}_{}.pkl'.format(date,seed_roi,seed_scoretype,seed_eff,post_scoretype,post_eff)
fpath = os.path.join(path,fname)

print('...saving to: ',fpath)
    
saveObject(fpath,data)

print('...Saved.')