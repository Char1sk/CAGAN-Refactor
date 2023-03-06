# train old in Kaggle
set -ex
python /kaggle/input/cagan1/CAGAN/train.py --log_name train_old01 --epochs 700 --batch_size 16 --test_start 100 --test_period 100 --save_models --vgg_model /kaggle/input/modelscagan/vgg.model --inception_model /kaggle/input/modelscagan/pt_inception.pth --data_folder /kaggle/input/cufs-cagan/CUFS-CAGAN
