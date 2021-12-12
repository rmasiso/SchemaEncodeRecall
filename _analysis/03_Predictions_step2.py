
from _functions import * #includes constants and score function
import sys
import deepdish as dd
import numpy as np
import os
import h5py
from scipy.stats import pearsonr

################
################ INPUTS
################

date = sys.argv[1] # 'date label to use'
hem = str(sys.argv[2]) # 'L' or 'R'
roi = sys.argv[3] # 'STS', 'SL', 'atlas'
roi_id = int(sys.argv[4]) # "9999" or int
scoretype = sys.argv[5] # "sl_percept_scores" or any directory name that makes sense
try:
    extra = sys.argv[6] #'_WithinSubj'/across modality
except:
    extra = ''

# datapath = '/jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/roi_percept_score'
# savepath = '/jukebox/norman/rmasis/clones/SchemaBigFiles/draft_PAPER/roi_predictions'
# fname = 


################
################ CONSTANTS 
################

nSubj = 30

### import rubric scores
rubric_scores = openObject('normalizedRubricScores_byStory.pkl')

################
################ FUNCTIONS
################

def GetStorySchemaScore(date,roi,roi_id,hem,scoretype,event='all'):


#     path = '../SchemaBigFiles/draft_PAPER/{}'.format(scoretype)
    
    path = '../../SchemaBigFiles/_PaperOutputData/{}'.format(scoretype)


    subj_selection = has_R if 'percept' in scoretype.split('_') else np.full(30,fill_value=True) #recalls only have 30 subj

    fname = '{date}_{roi}_{roi_id:04d}_{hem}_{scoretype}_{extra}.h5'.format(date=date, roi=roi, roi_id=roi_id,
                                                                hem=hem, scoretype=scoretype,extra=extra)

    score = {}
    for eff in ['story_effect','schema_effect']:
        score[eff] = dd.io.load(os.path.join(path,fname), '/' + hem + '/'+ eff )[subj_selection,:]
        
        if event=='all':
            score[eff] = np.nanmean(score[eff][:,:,:,0],axis=2) #average out the event dimension, select raw data (not permtuation)
        else:
            score[eff] = score[eff][:,:,event,0]
        
    return score

################
################ RUN ANALYSIS
################

### our 'X' in the regression
input_scores = GetStorySchemaScore(date,roi,roi_id,hem,scoretype,event='all')

# # # ## import the outcome variable of choice, the 'Y' in our regression
fname = 'normalizedRubricScores_byStory.pkl'
modeltype = 'predict_rubric' #for file saving purposes
outcome_scores = openObject(fname) 

# import the outcome variable of choice, the 'Y' in our regression
# fname = 'raw_wv_schema.pkl'
# modeltype = 'predict_raw_wv_schema' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression
# fname = 'raw_wv_story.pkl'
# modeltype = 'predict_raw_wv_story' #for file saving purposes
# outcome_scores = openObject(fname) 

# # # import the outcome variable of choice, the 'Y' in our regression
# fname = '0f_wv_schema.pkl'
# modeltype = 'predict_0f_wv_schema' #for file saving purposes
# outcome_scores = openObject(fname) 


# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = 'wv_euc_schema.pkl'
# modeltype = 'predict_wv_euc_schema' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = 'wv_euc_story.pkl'
# modeltype = 'predict_wv_euc_story' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = 'wv_gl_story.pkl'
# modeltype = 'predict_wv_gl_story' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = 'wv_gl_schema.pkl'
# modeltype = 'predict_wv_gl_schema' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = 'wv_bertsent_schema.pkl'
# modeltype = 'predict_wv_bertsent_schema' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = 'wv_bertsent_story.pkl'
# modeltype = 'predict_wv_bertsent_story' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "wv_bertsent_corr_schemadescrdiff.pkl"
# modeltype = 'predict_wv_bertsent_corr_schemadescrdiff' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "wv_bertsent_euc_schemadescrdiff.pkl"
# modeltype = 'predict_wv_bertsent_euc_schemadescrdiff' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "wv_bertsent_rawcos_schema.pkl"
# modeltype = 'predict_wv_bertsent_rawcos_schema' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "wv_bertsent_rawcos_story.pkl"
# modeltype = 'predict_wv_bertsent_rawcos_story' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "wv_fasttext_centered_euc_schemadescrdiff.pkl"
# modeltype = 'predict_wv_fasttext_centered_euc_schemadescrdiff' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "wv_fasttext_notcentered_euc_schemadescrdiff.pkl"
# modeltype = 'predict_wv_fasttext_notcentered_euc_schemadescrdiff' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "wv_fasttext_centered_corr_schemadescrdiff.pkl"
# modeltype = 'predict_wv_fasttext_centered_corr_schemadescrdiff' #for file saving purposes <--- official currently
# outcome_scores = openObject(fname) 


# # import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "loos_fasttext_loopca_schemadist.pkl"
# modeltype = 'predict_loos_fasttext_loopca_schemadist' #for file saving purposes
# outcome_scores = openObject(fname) 

### !!! last one used as of 20201018
# import the outcome variable of choice, the 'Y' in our regression // # new schema effect 20200805
# fname = "loos_fasttext_loopca_normrows_schemadist.pkl"
# modeltype = 'predict_loos_fasttext_loopca_normrows_schemadist' #for file saving purposes
# outcome_scores = openObject(fname) 

# # import the outcome variable of choice, the 'Y' in our regression // #20200812
# fname = 'wv_dist_schema.pkl'
# modeltype = 'predict_wv_dist_schema' #for file saving purposes
# outcome_scores = openObject(fname)

# import the outcome variable of choice, the 'Y' in our regression // #20200812
# fname = 'wv_fasttext_centered_cossim_schemadescrdiff.pkl'
# modeltype = 'predict_wv_fasttext_centered_cossim_schemadescrdiff' #for file saving purposes
# outcome_scores = openObject(fname)


### RUN PREDICTIONS

## regression arguments
fit_intercept = True ; remThresh = -1 ; alphaval = 0.00 #.5 #0.5 #1#5#0.1 #1 #5 #1.1 #.5 #1e-1 #1e-4 # 1e-9 

r2_scores = {}
coeffs = {}
for eff in ['story_effect','schema_effect']:
    r2_scores[eff] = np.zeros((nPerm+1))
    
    coeffs[eff] = np.zeros((nSubj,nPerm+1)) #store coefficients
    
    score_perm = input_scores[eff][:,:,np.newaxis].copy()

    for p in (range(nPerm+1)):
        
        print(p,end=' ') if p%100==0 else None
        
        if p == 0:
            ### MAKE predictions with raw data
            all_pred,all_true,all_baseline,test_r2s,regcoef,reginter = linear_reg(score_perm,
                                                                      outcome_scores,
                                                                      rubric_scores,
                                                                      fit_intercept=fit_intercept,
                                                                      remThresh = remThresh,
                                                                      alphain=alphaval,
                                                                        printr2=True)

            ### !!! kind of useless -- need to make sure that I check lowest common denominator for these
            x = all_pred[~np.isnan(all_pred)]
            y = all_true[~np.isnan(all_true)]
            base = all_baseline[~np.isnan(all_baseline)]

            r2 = get_r2(x,y,base)
            r2_scores[eff][p] = r2
            coeffs[eff][:,p] = regcoef # get back 30 coefficients for this regression for each 30 subj
            
#             coeffs[eff][p] = len(regcoef[regcoef>0])/ len(regcoef)
            
            print(regcoef)

        else:

            ### SHUFFLE STORIES
            score_perm = input_scores[eff][:,:,np.newaxis].copy()
            for s in range(nSubj):
                score_perm[s,:] = score_perm[s,np.random.permutation(nStories),:]


            ### MAKE PREDICTSONS with shuffled data
            all_pred,all_true,all_baseline,test_r2s,regcoef,reginter = linear_reg(score_perm,
                                                                      outcome_scores,
                                                                      rubric_scores,
                                                                      fit_intercept=fit_intercept,
                                                                      remThresh = remThresh,
                                                                      alphain=alphaval,
                                                                        printr2=True)
            x = all_pred[~np.isnan(all_pred)]
            y = all_true[~np.isnan(all_true)]
            base = all_baseline[~np.isnan(all_baseline)]

            r2 = get_r2(x,y,base)
            r2_scores[eff][p] = r2
            coeffs[eff][:,p] = regcoef 
#             coeffs[eff][p] = len(regcoef[regcoef>0])/ len(regcoef)
            print(regcoef)
            
            
#### correlate everything

corrs={}
for effect in ['story_effect','schema_effect']:
    x,y = nanxy(input_scores[effect].ravel(),outcome_scores.ravel()) #only look at valid values, some SLs might have nans for some subjects
    corrs[effect] = pearsonr(x,y)[0]
    
# corrs = {effect : pearsonr(input_scores[effect].ravel(), outcome_scores.ravel())[0] for effect in ['story_effect','schema_effect']}


################
################ SAVE DATA
################

# date = 20200712
# date = 20200715 #saving out correlations
# date = 20210315 #checking out cosine vs pearson
# date = 20200715 #using new mpfc clusters
# date = 20210909 #to get correct coefficient outputs for encoding vs retrieval <--- THIS IS NOW F'd up, doesnt contain coef of pure stuff, is now corresponding to 20210717 stuff
# date = 20210717 #corresponds to cr


# date= 20210929 #for original SLs with og data
# # date=20210930 #for within subject
# # date=20210931 #fro across modality
# date = 20210930 #within subj, but percept and recall scores are 2020626 with the "withinsubj" prefix

## USE THE DATE FROM THE INPUT FOR SAVING.

### make directory to save results if it doesnt already exist

path = '../../SchemaBigFiles/_PaperOutputData'

# if roi in ['mPFC', 'PMC', 'Aud', 'MTN', 'PM', 'MP','AT','STS','SFG']:
#     roitype = 'roi'
#     savedir = '{}_predictions'.format(scoretype)
# elif roi == 'atlas':
#     roitype = 'atlas'
#     savedir = '{}_predictions'.format(scoretype)
# elif roi.lower() == 'sl':
#     roitype='sl'
#     savedir = '{}_predictions'.format(scoretype)
    
path = os.path.join(path,'{}_predictions'.format(scoretype))
if not os.path.exists(path):
    print("...making directory: ", path)
    os.makedirs(path)    
else:
    print("...directory exists: ", path)
    
data_labels = ['story_effect','schema_effect']

fname = '{date}_{roi}_{roi_id:04d}_{hem}_{scoretype}_{modeltype}_{extra}.h5'.format(date=date, 
                                                                roi=roi, 
                                                                roi_id=roi_id,
                                                                hem=hem, 
                                                                scoretype=scoretype,
                                                                modeltype = modeltype,
                                                                extra = extra)
fullpath = os.path.join(path, fname)
print('...Saving to ', fullpath)

print("extra: ", extra)

with h5py.File(fullpath, 'w') as hf:

    # create dict entry
    hf = hf.create_group(hem)
    
    # create sub-dict entries
    for label in data_labels: 
        hf.create_dataset(label, data=r2_scores[label])
        
    ### save coeffs
    hf.create_dataset("story_coeff", data=coeffs['story_effect'])
    hf.create_dataset("schema_coeff", data=coeffs['schema_effect'])
    
    ##save corrs
    hf.create_dataset("story_corr", data=corrs['story_effect'])
    hf.create_dataset("schema_corr", data=corrs['schema_effect'])
        
print('...SAVING COMPLETE.')