cd /home/kyle/MedicalZooPytorch
git pull

for i in {1..10}
do
    echo $i
    nohup python train_unet_iseg19.py --fold_id $i > unet_iseg19_$i.nohup 2>&1
done


# rm ~/results/UNET3D/brats2019_1_11-08_02-51/ -r; rm unet_brats2019.nohup;git pull; nohup python train_unet_brats2019.py > unet_brats2019.nohup 2>&1 &
