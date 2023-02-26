## train
# train bg model
python tools/train_net.py -c configs/cat/bg.yaml
# inference bg model
python tools/test_net.py -c configs/cat/bg.yaml

# train fg model
python tools/train_net.py -c configs/cat/fg.yaml
# inference fg model
python tools/test_net.py -c configs/cat/fg.yaml

# train entire model
python tools/train_net.py -c configs/cat/joint_stage1.yaml
python tools/train_net.py -c configs/cat/joint_stage2.yaml
