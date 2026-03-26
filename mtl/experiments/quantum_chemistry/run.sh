mkdir -p ./save
mkdir -p ./trainlogs

gradient_method=dibsmtl
seed=1

nohup python trainer.py --gradient_method=$gradient_method --seed=$seed > trainlogs/dibsmtl-sd$seed.log 2>&1 &
