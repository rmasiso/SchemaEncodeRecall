################
################ LIBRARIES
################

from _functions import * #includes constants and score function
import sys
import deepdish as dd
import numpy as np
import os
import h5py

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd


################
################ FUNCTIONS
################

def GetStorySchemaScore(date,roi,roi_id,hem,scoretype,event='all'):


    
#     path = '../SchemaBigFiles/draft_PAPER/{}'.format(scoretype)
    
    path =  '../../SchemaBigFiles/_PaperOutputData/{}'.format(scoretype)


    subj_selection = has_R if 'percept' in scoretype.split('_') else np.full(30,fill_value=True) #recalls only have 30 subj

    fname = '{date}_{roi}_{roi_id:04d}_{hem}_{scoretype}.h5'.format(date=date, roi=roi, roi_id=roi_id,
                                                                hem=hem, scoretype=scoretype)

    score = {}
    for eff in ['story_effect','schema_effect']:
        score[eff] = dd.io.load(os.path.join(path,fname), '/' + hem + '/'+ eff )[subj_selection,:]
        
        if event=='all':
            score[eff] = np.nanmean(score[eff][:,:,:,0],axis=2) #average out the event dimension, select raw data (not permtuation)
        else:
            score[eff] = score[eff][:,:,event,0]
        
    return score


################
################ INPUTS
################

date = sys.argv[1] # 'date label to use'
hem = sys.argv[2] # 'L' or 'R'
roi = sys.argv[3] # 'STS', 'SL', 'atlas'
roi_id = int(sys.argv[4]) # "9999" or int
scoretype = sys.argv[5] # "sl_percept_scores" or any directory name that makes sense

### FOR DEBUGGING:
# date = 20200626 # 'date label to use'
# hem = 'R' # 'L' or 'R'
# roi = 'mPFC_k2_c0' # 'STS', 'SL', 'atlas'
# roi_id = 9999 #int(sys.argv[4]) # "9999" or int
# scoretype = 'roi_percept_score' #sys.argv[5] # "sl_percept_scores" or any directory name that makes sense


################
################ CONSTANTS 
################

nSubj = 30

### import rubric scores
rubric_scores = openObject('normalizedRubricScores_byStory.pkl')


################
################ SELECT INPUT AND OUTCOMES
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

################
################ RUN ANALYSIS
################

### make a dataframe that contains all my input information
data = pd.DataFrame(index=np.arange(nSubj*nStories), columns = ['subj','story_effect','schema_effect','outcome'])
data.subj = np.array([np.ones((nStories))*rep for rep in range(nSubj)]).reshape(nSubj*nStories).astype(int)
data.story_effect = input_scores['story_effect'].reshape(nSubj*nStories) # story regressor
data.schema_effect =  input_scores['schema_effect'].reshape(nSubj*nStories) # schema regressor
data.outcome = outcome_scores.reshape(nSubj*nStories) # outcome scores

data_perm = data.copy() ## where permutations will occur


#### COMPARITIVE STUFF 
benefits = {'story_benefit': np.zeros((2,nPerm+1)), 'schema_benefit': np.zeros((2,nPerm+1))}
# story_benefit = np.zeros((2,nPerm+1)) #m1 (schema) vs m2 (joint)
# schema_benefit = np.zeros((2,nPerm+1)) #m0 (story) vs m2 (joint)

for p in (range(nPerm+1)):
    
    if p>0:
        ### shuffle stories within subject
        data_perm['outcome'] = data.groupby("subj")['outcome'].transform(lambda x: x.sample(frac=1))

        
        ### shuffle everything
        #data_perm['outcome'] = outcome_scores.reshape(nSubj*nStories)[np.random.permutation(nSubj*nStories)

    m0 = ols('outcome ~ story_effect', data = data_perm).fit()
    m1 = ols('outcome ~ schema_effect', data = data_perm).fit()
    m2 = ols('outcome ~ story_effect + schema_effect', data = data_perm).fit()

    ## story benefit
    anova_results = anova_lm(m1,m2) #story benefit
    benefits['story_benefit'][0,p] = anova_results.at[1,'F']  # f-score contribution of story
    benefits['story_benefit'][1,p] = anova_results.at[1,'Pr(>F)'] # p-value

    ## schema benefit
    anova_results = anova_lm(m0,m2) #schema benefit
    benefits['schema_benefit'][0,p] = anova_results.at[1,'F']  # f-score contribution of story
    benefits['schema_benefit'][1,p] = anova_results.at[1,'Pr(>F)'] # p-value



################
################ SAVE DATA
################


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
    
path = os.path.join(path,'{}_model_comparisons'.format(scoretype))
if not os.path.exists(path):
    print("...making directory: ", path)
    os.makedirs(path)    
else:
    print("...directory exists: ", path)
    
data_labels = ['story_benefit','schema_benefit']

fname = '{date}_{roi}_{roi_id:04d}_{hem}_{scoretype}_{modeltype}.h5'.format(date=date, 
                                                                roi=roi, 
                                                                roi_id=roi_id,
                                                                hem=hem, 
                                                                scoretype=scoretype,
                                                                modeltype = modeltype)
fullpath = os.path.join(path, fname)
print('...Saving to ', fullpath)

with h5py.File(fullpath, 'w') as hf:

    # create dict entry
    hf = hf.create_group(hem)
    
    # create sub-dict entries
    for label in data_labels: 
        hf.create_dataset(label, data=benefits[label]) ## save benefits['story_benefit'] and benefits['schema_benefit']
        
#     ### save coeffs
#     hf.create_dataset("story_coeff", data=coeffs['story_effect'])
#     hf.create_dataset("schema_coeff", data=coeffs['schema_effect'])
    
#     ##save corrs
#     hf.create_dataset("story_corr", data=corrs['story_effect'])
#     hf.create_dataset("schema_corr", data=corrs['schema_effect'])
        
print('...SAVING COMPLETE.')
