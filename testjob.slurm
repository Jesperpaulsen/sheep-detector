#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50000
#SBATCH --job-name="summarization_pipeline_pegasus"
#SBATCH --output=output_summarization.out
#SBATCH --mail-user=jespergp@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
module purge
module load CUDA/11.1.1-GCC-10.2.0
module load Python/3.8.6-GCCcore-10.2.0

# Use if env not exist
#python -m venv env
source env/bin/activate

# Use if complains about pip version
# python -m pip install --upgrade pip
export PYTHONPATH="$HOME/master-thesis/env/bin"
python -V
pip -V
which python
which pip

#pip install --upgrade pip
#pip install numpy
#pip install matplotlib
#pip install python-dotenv
#python -m pip install -r requirements.txt
#cd models/sentence-embeddings/sent2vec 
#pip install .
#pip uninstall sent2vec -y
pip list

#python bio-multi-sum/main.py --embedding=sentence_bert --total_documents=5

python bio-multi-sum/main.py --embedding=biosentvec --total_documents=6 --docs_in_document_cluster=2 --pre_trained_model=pegasus
#python bio-multi-sum/main.py --embedding=sent2vec

echo "Script completed"
# Print all system information
uname -a
