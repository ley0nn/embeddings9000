#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --job-name=gpuCVWikimedia_reddit_general_ruby
#SBATCH --output=log/gpuCVWikimedia_reddit_general_ruby.log

module load Python/3.6.4-foss-2018a
module load Ruby/2.5.0-foss-2018a
python3 SVM_original.py -src '' -ftr embeddings -cls bilstm -mh5 CVWikimedia_reddit_general_ruby.h5 -tknzr CVWikimedia_reddit_general_ruby_tokenizer.pickle -trnp ../../4563973/toxicity_annotated_comments.tsv -ds wikimedia -pte ../../embeddings/reddit_general_ruby.txt -cln none -eps 10 -ptc 2 -lstmTrn False -lstmOp False -lstmTd False -lstmCV True
