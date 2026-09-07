$ConfigPath = Join-Path $PSScriptRoot 'config.yaml'
& 'E:\add1\envs\TCFormer1\python.exe' 'c:\Users\zyx19\Desktop\zhengliu\蒸馏\TCFormer\train_pipeline.py' --model tcformer --dataset bcic2a --gpu_id 0 --config $ConfigPath --interaug --max_epochs 1000
