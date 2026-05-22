#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 8000
#SBATCH --time 10:00:00

#SBATCH -o %j/log.out
#SBATCH -e %j/log.err
#SBATCH --mail-user torge.hinrichs@tuhh.de
#SBATCH --mail-type ALL

# Module initialisieren
. /etc/profile.d/module.sh
module purge
module load cuda/12.8


PYTHON=python3.11

TUHHID=cil9493

echo "Create directory on local disk"
WORKDIR=$SCRATCH/$SLURM_JOBID
INPUTDIR=/work/cps/$TUHHID/vultrition

export HF_HOME=$WORKDIR/hf_cache
export PIP_CACHE_DIR=$WORKDIR/pip_cache
export TRITON_HOME=$WORKDIR/triton
export HF_TOKEN="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"

mkdir -p $WORKDIR

rsync -r $INPUTDIR/* $WORKDIR/

cd $WORKDIR

echo "Python environment"
$PYTHON -m pip cache dir
$PYTHON -m venv env
source env/bin/activate
$PYTHON -m pip install --upgrade pip setuptools wheel 
$PYTHON -m pip install "torch>=2.7.1" "transformers>=4.53.0" numpy tqdm
$PYTHON -m pip install flash-attn --no-build-isolation

$PYTHON -m pip install -e .
vultrition -c examples/configs/primevul_paired.toml --run_analysis -o vultrition_results_primevul_paired.json

mkdir -p $INPUTDIR/new_results
cp $WORKDIR/vultrition_results_*.json $INPUTDIR/new_results/

rm -rf $WORKDIR

echo "Exit"
