import argparse
import jittor as jt
jt.set_global_seed(0)

jt.flags.use_cuda = 1

import utils
from models import DrivingForwardModel
from trainer import DrivingForwardTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--config_file', default ='./configs/nuscenes/main.yaml', type=str, help='config yaml file')
    parser.add_argument('--novel_view_mode', default='MF', type=str, help='MF of SF')
    args = parser.parse_args()
    return args

def train(cfg):    
    model = DrivingForwardModel(cfg, 0)
    trainer = DrivingForwardTrainer(cfg, 0)
    trainer.learn(model)

if __name__ == '__main__':
    args = parse_args()
    cfg = utils.get_config(args.config_file, mode='train', novel_view_mode=args.novel_view_mode)

    train(cfg)
