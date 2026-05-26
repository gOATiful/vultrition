#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-cpu 8000
#SBATCH --time 5:00:00

#SBATCH --job-name=primevul_paired

#SBATCH -o %j_%x/log.out
#SBATCH -e %j_%x/log.err
#SBATCH --mail-user torge.hinrichs@tuhh.de
#SBATCH --mail-type ALL

# Module initialisieren
. /etc/profile.d/module.sh
module purge
module load cuda/12.8

dataset="primevul"
variant="_paired"


PYTHON=python3.12

TUHHID=cil9493

echo "Create directory on local disk"
WORKDIR=$SCRATCH/$SLURM_JOBID_$dataset$variant
INPUTDIR=/work/cps/$TUHHID/vultrition

export HF_HOME=$WORKDIR/hf_cache
export PIP_CACHE_DIR=$WORKDIR/pip_cache
export TRITON_HOME=$WORKDIR/triton
export HF_TOKEN="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"

mkdir -p $WORKDIR
rsync -v -r $INPUTDIR/src $WORKDIR/
rsync -v -r $INPUTDIR/pyproject.toml $WORKDIR/
rsync -v -r $INPUTDIR/README.md $WORKDIR/
rsync -v -r $INPUTDIR/examples $WORKDIR/
rsync -v -r $INPUTDIR/datasets/$dataset $WORKDIR/datasets/


cd $WORKDIR

echo "Python environment"
$PYTHON -m pip cache dir
$PYTHON -m venv env
source env/bin/activate
$PYTHON -m pip install --upgrade pip wheel setuptools


$PYTHON -m pip install -e .
vultrition -c examples/configs/${dataset}${variant}.toml --run_analysis -o vultrition_results_$dataset$variant.json

mkdir -p $INPUTDIR/new_results
cp $WORKDIR/vultrition_results_*.json $INPUTDIR/new_results/
cp $WORKDIR/function_similarity_edges.csv $INPUTDIR/new_results/function_similarity_edges_$dataset${variant}.csv


rm -rf $WORKDIR

echo "Exit"
