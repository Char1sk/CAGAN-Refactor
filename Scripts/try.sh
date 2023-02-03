# try train.py in Local or Colab
set -ex
python train.py --log_name try --epochs 100 --batch_size 16 --test_start 20 --test_period 20
