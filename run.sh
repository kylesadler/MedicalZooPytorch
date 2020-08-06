cd /home/kyle/MedicalZooPytorch
git pull

for i in 1 2 3 4 5 6 7 8 9 10
do
    nohup python train_unet_iseg19.py --fold_id $1 > unet_iseg19_$1.nohup 2>&1;
done
