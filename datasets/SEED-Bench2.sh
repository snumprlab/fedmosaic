#!/bin/bash

# module load anaconda/3
# conda activate /home/mila/s/sparsha.mishra/scratch/federated
# ulimit -Sn $(ulimit -Hn)

huggingface-cli download AILab-CVC/SEED-Bench-2 --local-dir SEED-Bench-2 --cache-dir SEED-Bench-2 --repo-type dataset
cd SEED-Bench-2
unzip cc3m-image.zip
cat SEED-Bench-2-image.zip.* > SEED-Bench-2-image.zip
unzip SEED-Bench-2-image.zip
rm cc3m-image.zip
rm SEED-Bench-2-image.zip*