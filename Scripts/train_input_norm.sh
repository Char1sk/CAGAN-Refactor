# train in Kaggle, no BCE, normal params, INPUT NORM
set -ex
if [ -z "$1" ]; then name="train_input_norm"; else name=$1; fi;
python /kaggle/input/cagan1/CAGAN/train.py                                          \
    --log_name ${name}                                                              \
    --delta 1 --lamda 10 --gamma 5                                                  \
    --epochs 700 --batch_size 16                                                    \
    --test_start 100 --test_period 100 --save_models                                \
    --vgg_model /kaggle/input/modelscagan/vgg.model                                 \
    --inception_model /kaggle/input/modelscagan/pt_inception.pth                    \
    --data_folder /kaggle/input/cufs-cagan-new/CUFS-CAGAN-New
