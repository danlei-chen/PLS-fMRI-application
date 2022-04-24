#!/bin/tcsh
setenv PROJNAME $1
# setenv PROJNAME emoAvd_CSUSnegneu_trial 
# setenv PROJNAME painAvd_CSUS1snegneu_trial
# setenv PROJNAME combined_emoPainAvd_CSUSnegneu_trial
set FORMAT = (combined_trial_copes_neg combined_trial_copes_neu combined_trial_copes_negneu)
setenv FORMAT "$FORMAT"

setenv DATA /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled
setenv SCRIPTPATH /autofs/cluster/iaslab/users/danlei/FSMAP/scripts
setenv IMAGE /autofs/cluster/iaslab/users/jtheriault/singularity_images/jtnipyutil/jtnipyutil-2019-01-03-4cecb89cb1d9.simg
setenv SINGULARITY /usr/bin/singularity
setenv OUTPUT /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/$PROJNAME/connectivity_analyses/PLS

mkdir -p /scratch/$USER/$PROJNAME/wrkdir/
mkdir -p /scratch/$USER/$PROJNAME/output/
mkdir -p $OUTPUT

# rsync -r /autofs/cluster/iaslab/users/danlei/roi/selectedROI /scratch/$USER/$PROJNAME/wrkdir/
rsync /autofs/cluster/iaslab2/FSMAP/FSMAP_data/BIDS_modeled/emoAvd_CSUSnegneu_trial/connectivity_analyses/combined_trial_copes/sub-014_emo3_combined_trial_copes_neg.nii.gz /scratch/$USER/$PROJNAME/wrkdir/

# rsync -ra $SCRIPTPATH/model/search_region.nii /scratch/$USER/$PROJNAME/wrkdir/
rsync $SCRIPTPATH/model/1_run_SIMPLS/{run_SIMPLS_startup.sh,run_SIMPLS.py} /scratch/$USER/$PROJNAME/wrkdir/
rsync -r $SCRIPTPATH/model/1_run_SIMPLS/simpls /scratch/$USER/$PROJNAME/wrkdir/
chmod a+rwx /scratch/$USER/$PROJNAME/wrkdir/run_SIMPLS_startup.sh
cd /scratch/$USER

$SINGULARITY exec  \
--bind "$DATA/$PROJNAME/connectivity_analyses/PLS/extracted_data:/scratch/data" \
--bind "$OUTPUT/results/SIMPLS:/scratch/output" \
--bind "/autofs/cluster/iaslab/users/danlei/roi/PLSroi:/scratch/roi" \
--bind "/scratch/$USER/$PROJNAME/wrkdir:/scratch/wrkdir" \
$IMAGE \
/scratch/wrkdir/run_SIMPLS_startup.sh

# cd /autofs/cluster/iaslab/users/danlei/FSMAP/scripts/model/
# chmod -R a+rwx *

rm -r /scratch/$USER/$PROJNAME/
exit


# scp -r * /autofs/cluster/iaslab/users/danlei/test/
# scp dz609@door.nmr.mgh.harvard.edu:/autofs/cluster/iaslab/users/danlei/test/sub-014_emo3_combined_trial_copes_neg_smoothed3mm_masked.nii.gz /Users/chendanlei/Desktop/
