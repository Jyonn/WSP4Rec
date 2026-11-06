"""Evaluate explicitly completed submissions and create score-only feedback."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from wsprec.neutral_study import evaluate_round
from publish_neutral_feedback import main as publish

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,choices=[1,2],required=True);p.add_argument('keys',nargs='+');a=p.parse_args()
    torch.set_num_threads(1)
    for key in a.keys:evaluate_round(key,a.round)
    publish()
