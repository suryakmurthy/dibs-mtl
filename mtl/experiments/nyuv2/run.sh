mkdir -p ./save
mkdir -p ./trainlogs

loss_method=igbv1
gradient_method=nashmtl
seed=1

nohup python trainer.py --loss_method=$loss_method --gradient_method=$gradient_method --seed=$seed > trainlogs/igbv1-sd$seed.log 2>&1 &
