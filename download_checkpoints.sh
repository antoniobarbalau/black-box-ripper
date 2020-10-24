#!/bin/bash
set -e

gdown "https://drive.google.com/uc?id=1S_tI80dECZ1UC5FEktpBOGwinL4-0fRU"
mkdir -p checkpoints
mv models.zip checkpoints
cd checkpoints
unzip models.zip
rm -fr models.zip
