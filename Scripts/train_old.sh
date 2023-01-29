# debug train.py in Local or Colab
set -ex
python train.py --log_name train_old --epochs 700 --batch_size 32 --test_start 200 --test_period 100 --save_image_when_test --vgg_model /kaggle/input/modelscagan/vgg.model --inception_model /kaggle/input/modelscagan/pt_inception.pth --data_folder /kaggle/input/cufscaganchange/CUFS-CAGAN-Change/
