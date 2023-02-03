# debug train.py in Local or Colab
set -ex
python train.py --epochs 10 --batch_size 2 --test_start 5 --test_period 5
