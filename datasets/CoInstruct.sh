#!/bin/bash

# module load anaconda/3
# conda activate /home/mila/s/sparsha.mishra/scratch/federated
# ulimit -Sn $(ulimit -Hn)

huggingface-cli download q-future/Co-Instruct-DB --local-dir Co-Instruct-DB --repo-type dataset
cd Co-Instruct-DB
tar -xvf co-instruct-images.tar
