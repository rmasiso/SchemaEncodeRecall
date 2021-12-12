import numpy as np #for most things

# from util import * #for perm_groups

import pickle # for open and load functions
import nibabel as nib #for reading giftis

from sklearn import linear_model #for regression

import deepdish as dd #for loading up ROIlist
import os # for loading up ROIlist and file reading

###
### LOAD and SAVE  functions
###

def saveObject(strName, variable):
    with open(strName, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def openObject(strName):
    with open(strName, 'rb') as handle:
        objectOut = pickle.load(handle)
    return objectOut

def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :]
                      for da in gii.darrays]) 
    return data

###
### constants and variables
###

subjects = np.arange(30).astype(str) #subj 7 removed for recall portion due to technical issues; therefore 30 instead of 31


stories = ['Brazil',  'Derek', 'MrBean',   'PulpFiction',
           'BigBang', 'Santa', 'Shame',    'Vinny',
           'DueDate', 'GLC',   'KandD',    'Nonstop',
           'Friends', 'HIMYM', 'Seinfeld', 'UpInTheAir']

nPerm = 1000
nv = 40962 # number of vertices in fsaverage6
nStories = 16 
modality = 'VVVVAAAAVVVVAAAA'
schema_modality = 'AAAABBBBCCCCDDDD'
sc_order = 'RRRRRRRRAAAAAAAA' # here its 16 stories askjdf 18 stories includes airport and restuarnt schema

## indexing to remove subject without recall data
has_R = np.array(
    [True, True, True, True, True, True, False, True, True, True,
     True, True, True, True, True, True, True, True, True, True,
     True, True, True, True, True, True, True, True, True, True, True],
    dtype=bool)

### import rubric scores
# rubric_scores = openObject('normalizedRubricScores_byStory.pkl')


###
###
###

def GetROIlist(roi):
    ''' makes a dict of lists {'L':list, 'R':list} where each item in the list is a list of verts'''
    
    roi_path = '../_data/ROIs'

    if roi == 'atlas':
        MMP = {} ; 
        for hem in ['L','R']:
             MMP[hem] = read_gifti(os.path.join(roi_path,'{}h.HCP-MMP1.fsaverage6.gii'.format(hem.lower())))[0] 

        nROIs = len(np.unique(MMP['L']))
        ROIlist = {'L':[],'R':[]}
        for hem in ['L','R']:
            for i in range(nROIs):
                ROIlist[hem].append(np.where(MMP[hem]==i)[0])

    elif roi.lower() == 'sl':
        ROIlist = dd.io.load(os.path.join(roi_path,'SLlist_c10.h5'))
        nROIs = len(ROIlist['L'])
        
    return ROIlist,nROIs


###
### MASK
###
MASK = np.zeros((nStories,nStories)) #single story schema score mask

for st in range(nStories):
    curr_schema = sc_order[st]
    
    train_ind = [sc==curr_schema for sc in sc_order]*(np.arange(nStories)!=st) 
    test_ind = np.invert(train_ind)*(np.arange(nStories)!=st) 
    
    MASK[st,st] = 1 ## diagonal squares
    MASK[st,train_ind] = 2 ## same schema (top left & bottom right quadrants)
    MASK[st,test_ind] = 3 ## diff schema|offdiag (bottom left & top right qudrant)
    
###
### Function to extract vertices from an ROI
###
    
    
def ExtractROI(roi,roi_id,hem):
    '''
    - roi is string with "SL", "atlas", or specific ROI like "PMC" found in the /ROIs
    - roi_id is index of type integer. for searchlights this can range from 0-1483, for the atlas from 0-180, and if specific a priori ROI like "mPFC" or "Ang", this will just be a constant int of value 9999
    - returns a boolean array of the size of the ROI
    '''

    nv = 40962
    roi_verts = np.zeros((nv),dtype=bool) #create full hemisphere

    hemi = 'left' if hem == 'L' else 'None' if hem == 'None' else 'right'

#     roi_path = '/jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/ROIs'
    
    roi_path = '../_data/ROIs/'

    if roi == 'SL':
        SL_indeces = dd.io.load(os.path.join(roi_path,'SLlist_c10.h5'), '/'+ hem )[roi_id]
        roi_verts[SL_indeces] = True

    elif roi == 'atlas':
        atlas = {}
        atlas[hem] = read_gifti(os.path.join(roi_path,'{}h.HCP-MMP1.fsaverage6.gii'.format(hem.lower())))[0] 
        roi_indeces = np.where(atlas[hem] == roi_id)[0]
        roi_verts[roi_indeces] = True

    else: #for specific ROIs like "PMC", "mPFC", or networks like 'MTN'
        
        if 'hippo' not in roi: #for all OTHER ROIs
            verts = dd.io.load(os.path.join(roi_path,'{}_verts.h5'.format(roi))) 
            roi_verts[verts[hemi]] = True
        else: #for hippo
            roi_verts = np.zeros((1289),dtype=bool)
            if roi == 'hippo':
                roi_verts = np.full(1289, fill_value=True) ##i know how many voxels in hippo
            elif roi =='ahippo':
                verts = dd.io.load(os.path.join(roi_path,'{}_verts.h5'.format('hippo')))['ahippo']
                roi_verts[verts] = True
            elif roi == 'phippo':
                verts = dd.io.load(os.path.join(roi_path,'{}_verts.h5'.format('hippo')))['phippo']
                roi_verts[verts] = True
        
    return roi_verts
    
###
### STORY AND SCHEMA EFFECT
###

## for permutations 
def charmatch(s, c):
    return np.array([c == x for x in s])

## for permutations
def perm_groups(labels):
    groups = ''.join(set(labels))
    perm = np.zeros(len(labels), dtype=np.int_)
    for g in groups:
        group_inds = np.where(charmatch(labels, g))[0]
        perm[group_inds] = np.random.permutation(group_inds)

    return perm
        
###
### STORY AND SCHEMA EFFECT
###

def GetStorySchemaEffect(smat,nPerm=nPerm,MASK=MASK,modality=modality,schema_modality=schema_modality,test_mod='within_modality',seed=None):
    '''
        -Calculate Story and Schema Effect from similarity matrix.
        -smat.shape should be (nStories, nStories, nSubj)
        
        THIS VERSION DOESNT ACCOUNT FOR TRANSPOSING.. DO WE STILL NEED THAT.
        THIS VERSION DOESNT ACCOUNT FOR REMEMBERED STORIES ONLY.. WOULD THIS BE SOMETHING THAT IS DONE BEFORE RUNNING THIS FUNCTION?
    '''
    
    ## 
    assert MASK.shape[0] == smat.shape[0] == len(modality) == len(schema_modality)
    
    nStories = smat.shape[0]
    nSubj = smat.shape[2]

    story_effect = np.zeros((nSubj,nStories,nPerm+1))
    schema_effect = np.zeros((nSubj,nStories,nPerm+1))
    
    # set a seed for reproducibility if provided in arguments
    np.random.seed(seed) if seed != None else None
    
    for s in (range(nSubj)):

        ## prepare 16 story x 16 story similiarity matrices for permutation
        smat_perm_story = smat[:,:,s] 
        smat_perm_schema = smat[:,:,s]

        for p in range(nPerm+1):
            for st in range(nStories):

#                 subset_mask = [modality[st]==mod for mod in modality]  # index into subsets of the 16 stories
                
                within_modality = [modality[st]==mod for mod in modality]
                across_modality = np.invert(within_modality)
                
                # index into subsets of the 16 stories
                subset_mask = within_modality if test_mod=='within_modality' else across_modality


                ### 
                ### STORY EFFECT 
                ###
                diag_square = smat_perm_story[st,st] # story in question
                same_schema_ind = np.logical_and(MASK[st,:]==2,subset_mask)
                avg_same_schema = (np.nanmean(smat_perm_story[st,same_schema_ind]) + 
                                  np.nanmean(smat_perm_story[same_schema_ind,st])) * (1/2)
                
                # get similarity measure for story effect
                story_effect[s,st,p] = diag_square - avg_same_schema

                #visualize
    #             x = np.zeros((16,16))
    #             x[st,same_schema_ind] = 1
    #             x[same_schema_ind,st] = 2
    #             plt.imshow(x);plt.show()

                ### 
                ### SCHEMA EFFECT 
                ###    

                ### SCORING WITH MODALITY SENSITIVTY FIX (feb 2020)
                same_schema_ind = np.logical_and(MASK[st,:]==2, subset_mask)
                diff_schema_ind = np.logical_and(MASK[st,:]==3, subset_mask)

                same_schema = (np.nanmean(smat_perm_schema[st,same_schema_ind]) + 
                               np.nanmean(smat_perm_schema[same_schema_ind,st])) * (1/2)

                diff_schema = (np.nanmean(smat_perm_schema[st,diff_schema_ind]) +
                               np.nanmean(smat_perm_schema[diff_schema_ind,st])) * (1/2)   
                
                # get similarity measure for schema effect
                schema_effect[s,st,p] = same_schema - diff_schema


            ### permute story effect
            #nextperm = perm_groups(schema_type) ## 20200214 used this one
            nextperm = perm_groups(schema_modality) ## 20200215 onwards, to account for the modality sensitivty
            smat_perm_story = smat[nextperm, :, s].copy()

            ### permute schema effect
            ## OLD METHOD:
            #             nextperm = np.random.permutation(len(stories))
            ### NEW METHOD of permutation with modality sensitivey fix (feb 2020)
            nextperm = perm_groups(modality)
            smat_perm_schema = smat[np.ix_(nextperm,nextperm)].copy()
            
    return story_effect,schema_effect




###
### REGRESSION STUFF
###
    
def get_r2(y_pred, y_true, base):
    return 1 - np.sum((y_pred-y_true)**2)/np.sum((base-y_true)**2)

def linear_reg(X,Y,rubric_scores,fit_intercept=False,remThresh=-1,alphain=0.5,nSubj=30,printr2=False):
    
    all_pred = np.full(Y.shape,fill_value=np.nan) #30x16
    all_true = np.full(all_pred.shape,fill_value=np.nan)
    all_baselines = np.full(all_pred.shape,fill_value=np.nan)

    test_r2s = np.zeros(nSubj)
    train_coeffs = np.zeros(nSubj)
    train_intercepts = np.zeros(nSubj)

    for loo in range(nSubj):

        ##
        ## TRAIN
        ##
        train_mask = np.arange(nSubj) != loo
        train_remSubj,train_remStory = np.where(rubric_scores[train_mask]>remThresh)
        x_train = X[train_mask,:][train_remSubj,train_remStory,:]
        y_train = Y[train_mask,:][train_remSubj,train_remStory][:,np.newaxis]
        
        ##
        ## TEST
        ##
        test_remStory =  np.where(rubric_scores[loo]>remThresh)
        x_test = X[loo,test_remStory].squeeze(0)
        y_test = Y[loo,test_remStory].squeeze(0)[:,np.newaxis]

        #print('x_train: {} | y_train: {} | x_test: {} | y_test: {}'.format(x_train.shape,y_train.shape,x_test.shape,y_test.shape))
        
        ##
        ## RUN REGRESSION
        ##
        reg = linear_model.Ridge(alpha=alphain, fit_intercept=fit_intercept, normalize=False)

        reg.fit(x_train,y_train) #fit model with trainin data
        y_pred = reg.predict(x_test)
        y_pred_baseline = np.ones(y_pred.shape)*y_train.mean()
        
        test_r2 = get_r2(y_pred,y_test,y_pred_baseline)

        all_pred[loo,test_remStory] = y_pred.squeeze()
        all_true[loo,test_remStory] = y_test.squeeze()
        all_baselines[loo,test_remStory] = y_pred_baseline.squeeze()
        
        #print("...LINREG | testr2 = {:.3f}".format(test_r2)) if printr2 else None #print test r2s
        
        if type(reg.intercept_) == type(0.0): #if float
            intercept = reg.intercept_
        else:
            intercept = reg.intercept_.squeeze()
            
        test_r2s[loo] = test_r2
        train_coeffs[loo] = reg.coef_.squeeze()
        train_intercepts[loo] = intercept

    return all_pred,all_true,all_baselines,test_r2s,train_coeffs,train_intercepts

###
### SIGNIFICANCE TESTING
### 

def nullZ(X):
    # Last dimension of X is nPerm+1, with real data at 0 element
    X_roll = np.rollaxis(X, len(X.shape)-1)
    means = X_roll[1:].mean(axis=0)
    std = X_roll[1:].std(axis=0)
    if len(X.shape) > 1:
        std[std==0] = np.nan
    Z = (X_roll[0] - means) / std
    return Z

def SLtoVox(D, SLlist, nv, zeronan=True):
    # D is dict of left,right, with N x arbitrary dims
    # SLlist is dict of left, right list of length N, with vertices for each SL

    Dvox = dict()
    Dcount = dict()
    for hem in ['L', 'R']:
        Dvox[hem] = np.zeros((nv,)+ D[hem].shape[1:])
        Dcount[hem] = np.zeros((nv,)+(1,)*len(D[hem].shape[1:]))
        for i in range(len(SLlist[hem])):
            Dvox[hem][SLlist[hem][i]] += D[hem][i]
            Dcount[hem][SLlist[hem][i]] += 1

        Dcount[hem][Dcount[hem] == 0] = np.nan
        Dvox[hem] = Dvox[hem] / Dcount[hem]

        if zeronan:
            Dvox[hem][np.isnan(Dvox[hem])] = 0

    return Dvox

def FDR_p_hem(p):
    p_cat = np.concatenate((p['L'], p['R']))
    valid_inds = np.logical_not(np.isnan(p_cat))
    q_cat = np.ones(p_cat.shape[0])
    q_cat[valid_inds] = FDR_p(p_cat[valid_inds])

    q = dict()
    q['L'] = q_cat[:p['L'].shape[0]]
    q['R'] = q_cat[p['L'].shape[0]:]

    return q

def FDR_p(pvals):
    # Port of AFNI mri_fdrize.c
    assert np.all(pvals>=0) and np.all(pvals<=1)
    pvals[pvals < np.finfo(np.float_).eps] = np.finfo(np.float_).eps
    pvals[pvals == 1] = 1-np.finfo(np.float_).eps
    n = pvals.shape[0]

    qvals = np.zeros((n))
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    qmin = 1.0
    for i in range(n-1,-1,-1):
        qval = (n * sorted_pvals[i])/(i+1)
        if qval > qmin:
            qval = qmin
        else:
            qmin = qval
        qvals[sorted_ind[i]] = qval

    # Estimate number of true positives m1 and adjust q
    if n >= 233:
        phist = np.histogram(pvals, bins=20, range=(0, 1))[0]
        sorted_phist = np.sort(phist[3:19])
        if np.sum(sorted_phist) >= 160:
            median4 = n - 20*np.dot(np.array([1, 2, 2, 1]), sorted_phist[6:10])/6
            median6 = n - 20*np.dot(np.array([1, 2, 2, 2, 2, 1]), sorted_phist[5:11])/10
            m1 = min(median4, median6)

            qfac = (n - m1)/n
            if qfac < 0.5:
                qfac = 0.25 + qfac**2
            qvals *= qfac

    return qvals

def NonparametricP(dd_vox,sided=1):
    '''one or two-tailed non-parametric p-value calculation'''
    p_brain = np.zeros((nv))
    nPerm = dd_vox.shape[1] - 1 #number of permtutations
    ### find proportion of vertices greater than value of current vertex
    for v in range(nv): 
        if ~np.isnan(dd_vox[v,0]): 
            
            thesum = np.sum(np.abs(dd_vox[v,:])>=np.abs(dd_vox[v,0])) if sided==2 else np.sum((dd_vox[v,:])>=(dd_vox[v,0]))
                
#                 thesum = len(np.where(dd_vox[v,1:]>=(dd_vox[v,0]))[0])  #np.abs ensures a two tailed test.
#     #             thesum = np.sum(np.abs(dd_vox[v,1:])>=np.abs(dd_vox[v,0])) #np.abs ensures a two tailed test.
    

            p_brain[v] = (thesum/(nPerm+1))  #turn to fraction
        else:
            p_brain[v] = np.nan
    return p_brain

def nanxy(x,y):
    nans = np.logical_or(np.isnan(x),np.isnan(y))
    x = x[~nans]
    y = y[~nans]
    return x,y


####
####
####

