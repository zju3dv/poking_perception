## Setup data

Download data from [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/3150105275_zju_edu_cn/Eld_jMloZSRAir92yqhXwlEBzqvjY2YB5DUghs9x_gBkkg?e=wonrw7) and organize the project structure as follows:
```bash
pokingperception/ # project root
├─ data
│  ├─ kinect
├─ models
```


## Test with trained model

As an example, we describe steps to run the evaluation for the cat. Other objects are similar.

```bash
sh scripts/test_with_trained_model.sh
```

## Training

As an example, we describe steps to run the training for the cat. Other objects are similar.

You can directly run the commands as follows:

```bash
sh scripts/train.sh
```

If you want to re-compute the initialized object poses, follow the next steps:

```bash
# 1. Download RAFT checkpoint.
mkdir -p models/raft && gdown 1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O models/raft/
# 2. Perform optical flow estimation using RAFT.
python tools/kinect_robot/raft_inf_flow.py
# 3. Perform MaskFusion to initialize object poses and copy the output pose into the config files.
python tools/test_net.py -c configs/cat/maskfusion.yaml
# 4. Run the training commands in scripts/train.sh
```

