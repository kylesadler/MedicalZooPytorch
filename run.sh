cd /home/kyle/MedicalZooPytorch
git pull

for i in {1..10}
do
    echo $i
    nohup python train_unet_iseg19.py --fold_id $i > unet_iseg19_$i.nohup 2>&1 &;
done
