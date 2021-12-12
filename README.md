# Schema representations in distinct brain networks support narrative memory during encoding and retrieval
Code for the paper Masís-Obando, Norman, & Baldassano (submitted) Schema representations in distinct brain networks support narrative memory during encoding and retrieval

## Project Organization
--- insert file tree here

## Data

Both encoding and recall neural data in BIDS format is available at [OpenNeuro](https://openneuro.org/datasets/ds001510).

## Preprocessing

After downloading both the encoding and recall data, preprocess and obtain surface data in fsaverage6 resolution. 

### General notes on parallel processing

To run the searchlights and specific ROI analyses, the SLURM scheduler (on servers mantained by the Princeton Neuroscience Institute) was used. Scripts and commands (e.g., “sbatch”) in this repository include the necessary formulations to work on SLURM.

In some of the scripts below, ```module load pyger``` was used to instantiate a virtual environment with python 3.6 installed.

Because the date used to generate the code was 20211014, you will find that the commands in this repository contain this date.

Run code below through instructions chronologically. 


Hippocampus…

The main idea behind these analyses involves first learning within ROIs

## Main Analyses
### Figure 1: Memory Scores

Run the jupyter notebook ```00_MemoryPerformance_Fig1.ipynb```  to generate the file ```normalizedRubricScores_byStory.pkl``` which contains the quantified memory performance for each story for each subject. We will need this later for relating the neural scores to behavior. 
### Figure 2: story and schema scores for searchlights and specific a priori ROIs 



#### -- ENCODING -- 

**Searchlights**
```
sbatch 02_PerceptionStorySchema_SLs_step1.sh 20211014 L within_modality
sbatch 02_PerceptionStorySchema_SLs_step1.sh 20211014 R within_modality
```

**ROIs**
```
sh 02_PerceptionStorySchema_ROIs_step0.sh 20211014 within_modality
```

#### -- REINSTATEMENT --

**Searchlights**
```
sbatch 02_RecallStorySchema_SLs_step1.sh 20211014 L within_modality
sbatch 02_RecallStorySchema_SLs_step1.sh 20211014 R within_modality
```
**ROIs**
```
sh  02_RecallStorySchema_ROIs_step0.sh 20211014 within_modality
```

#### Analysis

**Searchlight ANALYSIS** 
1. Open the jupyter notebook labeled: ```_ImportSearchlightsGenerateMaps.ipynb```
2. Find the cell Labeled **Figure 2**
3. Run to save brainmaps in desired output directory

**ROI ANALYSIS** 
1. Open the jupyter notebook labeled: ``` _ImportROIsSeeResults.ipynb```
2. Find the cell labeled **Figure 2**
3. Run to output violin plots 

### Figure 3: Predicting behavioral memory for story details with neural measures from encoding and recall. 

#### SEARCHLIGHTS

**ENCODING** 
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_percept_score within_modality
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_percept_score within_modality
```

**REINSTATEMENT**
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_recall_score within_modality
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_recall_score within_modality
```

#### ROIs

**For both encoding and reinstatement:**
In ```03_Predictions_ROIs_step0.sh```, make sure that ```rois=( PMC mPFC PHC Ang SFG )```

```sh 03_Predictions_ROIs_step0.sh 20211014 within_modality``

#### Analysis


**Searchlight ANALYSIS**
1. Open the jupyter notebook labeled: ```_ImportSearchlightsGenerateMaps.ipynb```
2. Find the cell Labeled **Figure 3**
3. Run to save brainmaps in desired output directory

**ROI ANALYSIS**
1. Open the jupyter notebook labeled: ``` _ImportROIsSeeResults.ipynb```
2. Find the cell labeled **Figure 3**
3. Run to output violin plots 

### Figure 4. Regions with significant schema effects and positive associations with subsequent memory.

#### SEARCHLIGHTS

To identify the benefit of including schematic information at encoding or recall in addition to story information run:

``sh 03S2_SchemaBenefitModelComparisons_step1.sh 2021014``

#### Analysis

**SEARCHLIGHT**
In the notebook titled ``ImportSearchlightsGenerateMaps`` find the cell labeled ``Figure 4`` and run the cells under it.
Files will be saved in the default or (your desired) directory.

### Figure5. Mediation analysis with mPFC subclusters.

#### (1) Generate Clusters

First we have to run the silhouette analysis and save out the clusters.
Go to the jupyter notebook: ``05_Kmeans.ipynb`` and run all the cells.
The clusters will be saved into`` /_data/`` as ``mPFC_c0.h5`` and ``mPFC_c1.h5`` 

#### (2) Create Encoding and Reinstatement Story and Schema Scores

Run the encoding and reinstatement story and schema scores:

**ENCODING**

Open ``02_PerceptionStorySchema_ROIs_step0.sh`` and uncomment the second “rois” line argument that contains ``rois=( mPFC_k2_c0 mPFC_k2_c1 )``
Run in terminal with: ``sh 02_PerceptionStorySchema_ROIs_step0.sh 20211014 within_modality``

**REINSTATEMENT**

Open ``02_RecallStorySchema_ROIs_step0.sh`` and uncomment the second “rois” line argument that contains ``rois=( mPFC_k2_c0 mPFC_k2_c1 )``
Run in terminal with: ``sh 02_RecallStorySchema_ROIs_step0.sh 20211014 within_modality``

#### (3) Use scores in step 2 to predict behavior

Open ``03_Predictions_ROIs_step0.sh`` and uncomment the second “rois” line argument that contains ``rois=( mPFC_k2_c0 mPFC_k2_c1 )``
Run in terminal with: ``sh 03_Predictions_ROIs_step0.sh 20211014 within_modality``

#### (4) Visualize post-hoc mPFC subclusters results

Open ``_ImportROIsSeeResults.ipynb``
Find the section titled: ``Figure5; mPFC cluster results``
Run the cells to output the bar graphs.

#### (4) Run Mediation Analysis

Open the jupyter notebook ``05_Mediation.ipynb``
Run all the cells




















## Supplementary Analyses
### Figure 2 - Supp 1 (Encoding and Reinstatement Story and Schema Within-Subjects)

#### -- ENCODING -- 

**Searchlights**
```
sbatch 02S1_PerceptionStorySchema_WithinSubj_SLs_step1.sh 20211014 L 
sbatch 02S1_PerceptionStorySchema_WithinSubj_SLs_step1.sh 20211014 R
```

**ROIs**
```
sh 02S1_PerceptionStorySchema_WithinSubj_ROIs_step0.sh 20211014 
```

#### -- REINSTATEMENT --

**Searchlights**
```
sbatch 02S1_RecallStorySchema_WithinSubj_SLs_step1.sh 20211014 L
sbatch 02S1_RecallStorySchema_WithinSubj_SLs_step1.sh 20211014 R

```
**ROIs**
```
sh 02S1_RecallStorySchema_WithinSubj_ROIs_step0.sh 20211014
```

#### Analysis

**Searchlight ANALYSIS** 
1. Open the jupyter notebook labeled: ``_ ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell Labeled **Figure 2**
3. Uncomment the line ``extra = within_modality_within_subj``
4. Run to save brainmaps in desired output directory

**ROI ANALYSIS** 
1. Open the jupyter notebook labeled:  ``_ImportROIsSeeResults.ipynb``
2. Find the cell labeled **Figure 2**
3. Uncomment the line  ``extra = within_modality_within_subj``
4. Run to output violin plots 

### Figure 2 - Supp 2 (Encoding and Reinstatement Story and Schema Across-Modality)

========ACROSS MODALITY========

#### ENCODING

**Searchlights**
```
sbatch 02_PerceptionStorySchema_SLs_step1.sh 20211014 L across_modality
sbatch 02_PerceptionStorySchema_SLs_step1.sh 20211014 R across_modality
```

**ROIs**
In ``02_PerceptionStorySchema_ROIs_step0.sh``, make sure that ``rois=( PMC mPFC PHC Ang SFG )``

``sh 02_PerceptionStorySchema_ROIs_step0.sh 20211014 across_modality``


#### REINSTATEMENT

**Searchlights**
```
sbatch 02_RecallStorySchema_SLs_step1.sh 20211014 L across_modality
sbatch 02_RecallStorySchema_SLs_step1.sh 20211014 R across_modality
```
**ROIs**
In ``02_RecallStorySchema_ROIs_step0.sh``, make sure that ``rois=( PMC mPFC PHC Ang SFG )``

``sh  02_RecallStorySchema_ROIs_step0.sh 20211014 across_modality``

#### Analysis

**Searchlight ANALYSIS** 
1. Open the jupyter notebook labeled: ``_ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell Labeled **Figure 2**
3. Uncomment the line ``extra = ‘across_modality’ ``
4. Run to save brainmaps in desired output directory

**ROI ANALYSIS**
1. Open the jupyter notebook labeled: ``_ImportROIsSeeResults.ipynb``
2. Find the cell labeled **Figure 2**
3. Uncomment the line ``extra = ‘across_modality’``
4. Run to output violin plots 


### Figure 3–Figure supplement 1. Predicting PMC reinstatement story scores with encoding schema scores across cortex.

To find associations with PMC’s reinstatement story score and the rest of the brain’s encoding schema score, run the following line:

```
sbatch 03S1_SeedtoSL_step1.sh 20211014 PMC roi_recall_score story_effect SL sl_percept_score schema_effect
```

A brainmap will be saved in the default directory `` ‘../../SchemaBigFiles/_PaperOutputData/Brainmaps’ ``

 $date $seed_roi $seed_scoretype $seed_eff $post_roi $post_scoretype $post_eff

### Figure 3–Figure supplement 2. Behavioral memory benefit of neural schema scores

#### Searchlights

```
sh 03S2_SchemaBenefitModelComparisons_step1.sh 
```

Saves searchlight results under a folder called ``sl_percept_score_model_comparisons`` and ``sl_recall_score_model_comparisons`` for the benefit analysis at encoding and retrieval, respectively.

#### Analysis:
1. Find the cell labeled: ``Figure 3 - Supplement 2``
2. Run the cell 
3. Brainmap will be saved in ``../../SchemaBigFiles/_PaperOutputData`` under ``model comparisons``

### Figure 3–Figure supplement 3. Predicting behavioral memory for story details with neural measures from encoding and recall within-subjects.

#### SEARCHLIGHTS

**ENCODING **
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_percept_score within_modality_within_subj
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_percept_score within_modality_within_subj
```

**REINSTATEMENT**
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_recall_score within_modality_within_subj
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_recall_score within_modality_within_subj
```

#### ROIs

**For both encoding and reinstatement:**
``sh 03_Predictions_ROIs_step0.sh 20211014 within_modality_within_subj``

#### Analysis

**Searchlight ANALYSIS**
1. Open the jupyter notebook labeled: ``_ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell Labeled **Figure 3**
3. Uncomment the line ``extra = 'within_modality_within_subj'``
4. Run to save brainmaps in desired output directory

**ROI ANALYSIS**
1. Open the jupyter notebook labeled:  ``_ImportROIsSeeResults.ipynb``
2. Find the cell labeled **Figure 3**
3. Uncomment the line ``extra = 'within_modality_within_subj'``
4. Run to output violin plots 

### Figure 3–Figure supplement 4. Predicting behavioral memory for story details with neural measures from encoding and recall across-modality.


#### SEARCHLIGHTS

**ENCODING **
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_percept_score across_modality
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_percept_score across_modality
```

**REINSTATEMENT**
```
sbatch 03_Predictions_SLs_step1.sh 20211014 L sl_recall_score across_modality
sbatch 03_Predictions_SLs_step1.sh 20211014 R sl_recall_score across_modality
```

#### ROIs

**For both encoding and reinstatement:**
In ``03_Predictions_ROIs_step0.sh``, make sure that ``rois=( PMC mPFC PHC Ang SFG )``

``sh 03_Predictions_ROIs_step0.sh 20211014 across_modality``

sbatch 03_Predictions_ROIs_step1.sh 20211014 None hippo roi_recall_score across_modality

#### Analysis

**Searchlight ANALYSIS**
1. Open the jupyter notebook labeled: ``_ImportSearchlightsGenerateMaps.ipynb``
2. Find the cell Labeled **Figure 3**
3. Uncomment the line ``extra = 'across_modality'``
4. Run to save brainmaps in desired output directory

**ROI ANALYSIS** 
1. Open the jupyter notebook labeled: ``_ImportROIsSeeResults.ipynb``
2. Find the cell labeled **Figure 3**
3. Uncomment the line ``extra = 'across_modality'``
4. Run to output violin plots 

