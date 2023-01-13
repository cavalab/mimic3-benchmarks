rdir="results_archive_dim100_22-06-07"
ntrials=10
seeds=$(cat seeds.txt | head -n $ntrials)
mkdir -p $rdir

phenotype_names=(
'Acute-and-unspecified-renal-failure'
'Acute-cerebrovascular-disease'
'Acute-myocardial-infarction'
'Cardiac-dysrhythmias'
'Chronic-kidney-disease'
'Chronic-obstructive-pulmonary-disease-and-bronchiectasis'
'Complications-of-surgical-procedures-or-medical-care'
'Conduction-disorders'
'Congestive-heart-failure-nonhypertensive'
'Coronary-atherosclerosis-and-other-heart-disease'
'Diabetes-mellitus-with-complications'
'Diabetes-mellitus-without-complication'
'Disorders-of-lipid-metabolism'
'Essential-hypertension'
'Fluid-and-electrolyte-disorders'
'Gastrointestinal-hemorrhage'
'Hypertension-with-complications-and-secondary-hypertension'
'Other-liver-diseases'
'Other-lower-respiratory-disease'
'Other-upper-respiratory-disease'
'Pleurisy-pneumothorax-pulmonary-collapse'
'Pneumonia-(except-that-caused-by-tuberculosis-or-sexually-transmitted-disease)'
'Respiratory-failure-insufficiency-arrest-(adult)'
'Septicemia-(except-in-labor)'
'Shock'
)

# Job parameters
# cores
N=1
q="epistasis_long"
mem=16384

timeout=36000
count=0

for s in ${seeds[@]} ; do
    for p in ${phenotype_names[@]} ; do
        job_name="${p}_feat_seed${s}" 
        job_file="${rdir}/${job_name}" 

        echo python -m mimic3models.phenotyping.feat.main  --features tsfresh --output_dir $rdir --phenotype ""$p""
        bsub -o "${job_file}.out" \
             -n $N \
             -J $job_name \
             -q $q \
             -R "span[hosts=1] rusage[mem=${mem}]" \
             -W $timeout \
             -M $mem \
             python -m mimic3models.phenotyping.feat.main  \
                --features tsfresh \
                --output_dir $rdir \
                --phenotype ""$p"" \
                --seed $s

        ((++count))
    done
done

echo "submitted $count jobs."
