#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=gpuCVWikimedia_glove
#SBATCH --output=log/gpuCVWikimedia_glove.log

module load Python/3.6.4-foss-2018a
module load Ruby/2.5.0-foss-2018a
python3 SVM_original.py -src '' -ftr embeddings -cls bilstm -mh5 CVWikimedia_glove.h5 -tknzr CVWikimedia_glove_tokenizer.pickle -trnp ../../4563973/toxicity_annotated_comments.tsv -ds wikimedia -pte ../../embeddings/glove.twitter.27B.200d.txt -cln none -eps 10 -ptc 2 -lstmTrn False -lstmOp False -lstmTd False -lstmCV True
