#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python3 '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/PLS/scripts/1_run_PLS/run_PLS_local.py'

"""
Created on Tue Mar 23 16:07:42 2021
@author: Danlei

SIMPLS method here was built based on rmarkello's pyls package: 
    - source code: 
        - https://github.com/rmarkello/pyls
        - https://github.com/rmarkello/pyls/blob/master/pyls/types/regression.py
    - user guide:
        - https://pyls.readthedocs.io/en/latest/index.html
        - although most of the guide doesn't work for me, so adapted here
    - the code used in the script has been verified comparatively to MATLAB's plsregress result (not open sourced)
        - https://www.mathworks.com/help/stats/plsregress.html#d123e729617
        - https://www.mathworks.com/help/stats/partial-least-squares-regression-and-principal-components-regression.html
SIMPLS conceptual idea: https://learnche.org/pid/latent-variable-modelling/projection-to-latent-structures/conceptual-mathematical-and-geometric-interpretation-of-pls
Comparatively, SIMPLS can be contrasted with sklearn's method
     - user guide: 
        - https://docs.w3cub.com/scikit_learn/modules/generated/sklearn.cross_decomposition.plsregression
        - https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition
     - source code: 
        - https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/cross_decomposition/pls_.py#L590

Other resources as backup:
#PLS function that takes 2 datasets, outputs coefficients 
fNIR data PLS example: https://nirpyresearch.com/partial-least-squares-regression-python/
PCR v PLS: https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py
conventional correlation example :https://nilearn.github.io/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html#sphx-glr-auto-examples-03-connectivity-plot-seed-to-voxel-correlation-py
"""

import pandas as pd
import nibabel as nib
import numpy as np
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import os
import glob
from nilearn.image import resample_img
from nilearn import plotting
from nilearn.image import mean_img
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
import statistics 
from datetime import date
import math
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict  
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy import stats, linalg
import sys
import pickle
sys.path.insert(1, '/scratch/wrkdir/simpls/')
import simpls

def optimise_SIMPLS_cv(X, y, label, k_folds=3, n_comp=10, seed_roi=None, target_roi=None, output_dir=None, plot_components=True):        

    component = np.arange(1, n_comp+1)   
    group_kfold = GroupKFold(n_splits=k_folds)
    
    component_selection_df = pd.DataFrame(columns = ['mse', 'varexp', 'component', 'optimal'])
    for i in component:
        print('\rworking on component '+str(i)+' out of '+str(n_comp), end="")
        
        for train_index, test_index in group_kfold.split(X, y, label):
            # define training and testing set
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            
            # Fit to the training set  
            simpls_cv = simpls.pls_regression(X_train, y_train, n_components=i, n_perm=0, n_boot=0)
            beta_cv = simpls_cv.x_weights @ simpls_cv.y_loadings.T
            beta_cv = np.row_stack([y_train.mean(axis=0) - (X_train.mean(axis=0) @ beta_cv), beta_cv])
            
            #test on testing set
            #based on https://www.mathworks.com/help/stats/plsregress.html#d123e729617: yfit = [ones(size(X,1),1) X]*beta;
            y_predict_cv = np.append(np.tile([[1]], (X_test.shape[0],1)), X_test, axis=1) @ beta_cv
            
            #get mean squared error between predicted and real data
            mse = mean_squared_error(y_test, y_predict_cv)
            component_selection_df= component_selection_df.append({'mse':mse, 'varexp':simpls_cv.varexp[-1], 'component':int(i), 'optimal':0}, ignore_index=True)
        # y_cv = cross_val_predict(pls_cv, X, y, cv=3)            
        # mse[i].append(mean_squared_error(y, y_cv))    
        # correlation[i].append(np.corrcoef(pls_cv.x_scores_.T[-1,:], pls.y_scores_.T[-1,:])[0,1])
    print(' ')
        
    #find optimal component
    #find average across components
    mse_average = component_selection_df.groupby('component', as_index=False)['mse'].mean()
    varexp_average = component_selection_df.groupby('component', as_index=False)['varexp'].mean()
    #optinal number of components from minumum mse
    optimal_comp = mse_average.loc[np.argmin(mse_average['mse'])]['component']
    component_selection_df['optimal'][component_selection_df['component']==optimal_comp] = 1
    print("Suggested number of components: ", optimal_comp)  
    if output_dir and seed_roi and target_roi:
        component_selection_df.to_csv(os.path.join(output_dir, 'mse_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(label)))+'.tsv'))
    
    if plot_components:
        fig, axs = plt.subplots(1,2,figsize=(12, 3))
        axs[0].plot(component, np.array(mse_average['mse']), '-o', ms = 5)
        axs[0].plot(optimal_comp, mse_average['mse'].loc[mse_average['component']==optimal_comp], 'o', ms = 8, mfc='red')    
        axs[0].hlines(y=mse_average['mse'].loc[mse_average['component']==optimal_comp], xmin=1, xmax=n_comp, color='r', linestyle='--', alpha=0.5)
        axs[0].set_xlabel('Number of PLS components')              
        axs[0].set_ylabel('MSE')     
        axs[1].plot(component, np.cumsum(np.array(varexp_average['varexp'])), '-o', ms = 5)
        axs[1].plot(optimal_comp, np.cumsum(np.array(varexp_average['varexp']))[int(optimal_comp-1)], 'o', ms = 8, mfc='red')     
        axs[1].hlines(y=np.cumsum(np.array(varexp_average['varexp']))[int(optimal_comp-1)], xmin=1, xmax=n_comp, color='r', linestyle='--', alpha=0.5)
        axs[1].set_xlabel('Number of PLS components')              
        axs[1].set_ylabel('Cumulative variance explained') 
        if seed_roi and target_roi:         
            plt.title('PLS - seed: '+seed_roi+'; target:'+target_roi+'; N:'+str(len(set(label))))   
        if output_dir and seed_roi and target_roi:
            plt.savefig(os.path.join(output_dir, seed_roi+'_to_'+target_roi+'_N'+str(len(set(label)))+'.png'))
        # plt.show()
        plt.close(fig)
    
    # Define PLS object with optimal number of components and save prediction using entire seed data
    # simpls_opt = simpls.pls_regression(X, y, n_components=optimal_comp, n_perm=1000, n_boot=1000)  
    simpls_opt = simpls.pls_regression(X, y, n_components=optimal_comp, n_perm=0, n_boot=0)  
    beta_opt = simpls_opt.x_weights @ simpls_opt.y_loadings.T
    beta_opt = np.row_stack([y.mean(axis=0) - (X.mean(axis=0) @ beta_opt), beta_opt])
    y_predict_simpls_opt = np.append(np.tile([[1]], (X.shape[0],1)), X, axis=1) @ beta_opt
    
    Z=linalg.pinv2(X) @ simpls_opt.y_scores #correspond to the same Z in Kragel paper
    V=linalg.pinv2(y) @ simpls_opt.x_scores #correspond to the same V in Kragel paper
        
    return simpls_opt, optimal_comp, y_predict_simpls_opt, Z, V, component_selection_df
 
#inputs
# projs = ['emoAvd_CSUSnegneu_trial', 'painAvd_CSUS1snegneu_trial']
# projs = ['emoAvd_CSUSnegneu_trial']
projs = [os.environ['PROJNAME']]
# data_format = 'combined_trial_copes_neg'
data_formats = os.environ.get("FORMAT").split(" ")
# seed_target_pairs = [('SC', 'PAG'), ('PAG', 'SC'), ('SC', 'SN'), ('SN', 'SC'), ('SC', 'VTA'), ('VTA', 'SC'), ('SC', 'Pulvinar'), ('Pulvinar', 'SC'), ('SC', 'Amygdala'), ('Amygdala', 'SC'), ('SC', 'Hippocampus'), ('Hippocampus', 'SC'), ('SC', 'Hypothalamus'), ('Hypothalamus', 'SC'), ('SC', 'Caudate'), ('Caudate', 'SC'), ('SC', 'Habenular'), ('Habenular', 'SC'), ('SC', 'Putamen'), ('Putamen', 'SC'), ('SC', 'PBN'), ('PBN', 'SC'), ('SC', 'Red'), ('Red', 'SC'), ('SC', 'LC'), ('LC', 'SC'), ('SC', 'VSM'), ('VSM', 'SC'), ('SC', 'Thalamus'), ('Thalamus', 'SC')]
# seed_target_pairs = [('SC', 'PAG'), ('PAG', 'SC'), ('SC', 'SN'), ('SN', 'SC'), ('SC', 'VTA'), ('VTA', 'SC'), ('SC', 'Pulvinar'), ('Pulvinar', 'SC'), ('SC', 'Amygdala'), ('Amygdala', 'SC'), ('SC', 'Hippocampus'), ('Hippocampus', 'SC')]
seed_target_pairs = [('SC', 'Caudate'), ('Caudate', 'SC'), ('SC', 'Habenular'), ('Habenular', 'SC'), ('SC', 'Putamen'), ('Putamen', 'SC'), ('SC', 'PBN'), ('PBN', 'SC'), ('SC', 'Red'), ('Red', 'SC'), ('SC', 'LC'), ('LC', 'SC'), ('SC', 'VSM'), ('VSM', 'SC'), ('SC', 'Thalamus'), ('Thalamus', 'SC')]
# seed_target_pairs = [('MGN', 'A1'), ('A1', 'MGN'), ('LG', 'V1'), ('V1', 'LG'), ('LG', 'A1'), ('A1', 'LG'), ('MGN', 'V1'), ('V1', 'MGN')]
# seed_target_pairs = [('MGN', 'A1'), ('A1', 'MGN'), ('LG', 'V1'), ('V1', 'LG')]
# seed_target_pairs = [('LG', 'A1'), ('A1', 'LG'), ('MGN', 'V1'), ('V1', 'MGN')]
n_comp = 50
k_folds = 5

# data_dir_base = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/PLS/results/extracted_data/'
# output_dir_base = '/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/PLS/results'
# roi_dir_list = glob.glob('/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/PLS/PLSroi/*')
# sample_subj_img = nib.load('/Volumes/GoogleDrive/My Drive/U01/AffPainTask_connectivity/analysis/connectivity/combined_trial_copes/emoAvd_CSUSnegneu_trial/sub-014_emo3_combined_trial_copes_neg.nii.gz')
data_dir_base = '/scratch/data/'
output_dir_base = '/scratch/output'
roi_dir_list = glob.glob('/scratch/roi/*')
sample_subj_img = nib.load('/scratch/wrkdir/sub-014_emo3_combined_trial_copes_neg.nii.gz')

for seed_target_pair in seed_target_pairs:
    seed_roi = seed_target_pair[0]
    target_roi = seed_target_pair[1]
    print('from '+seed_roi+' to '+target_roi)

    for proj in projs:
        print(proj)
        # data_dir = os.path.join(data_dir_base, proj)
        data_dir = data_dir_base

        for data_format in data_formats:
            print(data_format)
    
            ################################################
            ############# read combined data ###############
            # seed_roi = 'A1'
            # target_roi = 'V1'
            output_dir = os.path.join(output_dir_base, data_format, seed_roi+'_to_'+target_roi)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            #read in the saved data
            seed_file = glob.glob(os.path.join(data_dir, '*'+seed_roi+'*'+data_format+'.npy'))[0]
            seed_data = np.load(seed_file)
            target_file = glob.glob(os.path.join(data_dir, '*'+target_roi+'*'+data_format+'.npy'))[0]
            target_data = np.load(target_file)
            combined_event = pd.read_table(os.path.join(data_dir, 'event_file_'+data_format+'.tsv'), sep='\t', index_col=0)
        
            # #crop data
            # seed_res_combined_data = seed_res_combined_data[combined_event['subj']!='sub-037',:]
            # target_res_combined_data = target_res_combined_data[combined_event['subj']!='sub-037',:]
            # combined_event = combined_event.loc[combined_event['subj']!='sub-037']
            # seed_res_combined_data = seed_res_combined_data[combined_event['subj']!='sub-034',:]
            # target_res_combined_data = target_res_combined_data[combined_event['subj']!='sub-034',:]
            # combined_event = combined_event.loc[combined_event['subj']!='sub-034']
        
            subj_label = combined_event['subj']
            # run_session_label = combined_event['trial']
        
            print(seed_roi+" seed data size")
            print(seed_data.shape)
            print(target_roi+" target data size")
            print(target_data.shape)
            num_subj = len(set(subj_label))
            print("number of subjects")
            print(num_subj)
        
            #check if the file already exists, if so, skip    
            if len(glob.glob(output_dir+'/*.pkl')) == 0:

                if n_comp > seed_data.shape[1] or n_comp > target_data.shape[1]:
                    n_comp=np.min([seed_data.shape[1],  target_data.shape[1]])

                simpls_opt, optimal_comp, y_predict_simpls_opt, Z, V, _ = optimise_SIMPLS_cv(seed_data, target_data, subj_label,
                               k_folds, n_comp,
                               seed_roi, target_roi,
                               output_dir, plot_components=True)

                #save weights (x_weights is r in Phil's paper)
                pls_opt_x_weights = simpls_opt.x_weights
                np.save(os.path.join(output_dir, 'simpls_opt_x_weights_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(subj_label)))+'_comp'+str(optimal_comp)+'_data.npy'),pls_opt_x_weights)
                #save loadings
                pls_opt_y_loadings = simpls_opt.y_loadings
                np.save(os.path.join(output_dir, 'simpls_opt_y_loadings_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(subj_label)))+'_comp'+str(optimal_comp)+'_data.npy'),pls_opt_y_loadings)
                # #save scores (x_scores is t; y scores is u in Phil's paper)
                pls_opt_x_scores = simpls_opt.x_scores
                np.save(os.path.join(output_dir, 'simpls_opt_x_scores_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(subj_label)))+'_comp'+str(optimal_comp)+'_data.npy'),pls_opt_x_scores)
                pls_opt_y_scores = simpls_opt.y_scores
                np.save(os.path.join(output_dir, 'simpls_opt_y_scores_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(subj_label)))+'_comp'+str(optimal_comp)+'_data.npy'),pls_opt_y_scores)
                #save optimal prediction of target
                np.save(os.path.join(output_dir, 'simpls_opt_y_predicted_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(subj_label)))+'_comp'+str(optimal_comp)+'_data.npy'),y_predict_simpls_opt)
                #save transformed matrix based on each other's scores
                np.save(os.path.join(output_dir, 'simpls_opt_Z_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(subj_label)))+'_comp'+str(optimal_comp)+'_data.npy'),Z)
                np.save(os.path.join(output_dir, 'simpls_opt_V_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(subj_label)))+'_comp'+str(optimal_comp)+'_data.npy'),V)
          
                with open(os.path.join(output_dir, 'simpls_opt_'+seed_roi+'_to_'+target_roi+'.pkl'), 'wb') as f:
                    pickle.dump(simpls_opt, f, pickle.HIGHEST_PROTOCOL)
                
                # target_img = nib.load([i for i in roi_dir_list if target_roi in i][0])
                # target_img_resampled = resample_img(target_img,
                #             target_affine=sample_subj_img.affine,
                #             target_shape=sample_subj_img.shape[0:3],
                #             interpolation='nearest')
                # target_masker = NiftiMasker(mask_img=target_img_resampled, standardize=True)
                # target_masker.fit(sample_subj_img) 
            
                # #average optimal target region prediction across TR
                # y_predict_opt_df = combined_event.copy()
                # y_predict_opt_df = y_predict_opt_df.join(pd.DataFrame(y_predict_opt))
                # y_predict_opt_df_subj = y_predict_opt_df.groupby(['subj']).mean()
                # y_predict_opt_avg = np.mean(y_predict_opt_df_subj)
            
                # y_predict_opt_avg_data = target_masker.inverse_transform(y_predict_opt_avg).get_fdata()
                # y_predict_opt_avg_data[target_img.get_fdata()==0] = np.nan
                # y_predict_opt_avg_img = nib.Nifti1Image(y_predict_opt_avg_data, sample_subj_img.affine, sample_subj_img.header)
                # nib.save(y_predict_opt_avg_img, os.path.join(output_dir, 'predicted_pls_target_'+seed_roi+'_to_'+target_roi+'_N'+str(len(set(subj_label)))+'_comp'+str(optimal_comp)+'_data.nii.gz'))
            
                # #plot optimal component correlation and mse
                # correlation_opt = component_selection_df[component_selection_df['component']==optimal_comp]










    
    
    
    