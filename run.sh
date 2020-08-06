cd /home/kyle/MedicalZooPytorch
git pull

for i in {1..4}
do
    echo $i
    nohup python train_unet_iseg19.py --fold_id $i > unet_iseg19_$i.nohup 2>&1 &
done

nohup python train_unet_iseg19.py --fold_id 5 > unet_iseg19_5.nohup 2>&1

for i in {6..10}
do
    echo $i
    nohup python train_unet_iseg19.py --fold_id $i > unet_iseg19_$i.nohup 2>&1 &
done
