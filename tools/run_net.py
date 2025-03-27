#!/usr/bin/env python3

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_avgaze_net import test
from train_avgaze_net import train

import wandb

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    use_wandb = args.use_wandb
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    

    print("use_wandb in main : ", use_wandb)

    if use_wandb:
        run = wandb.init(project="avgaze")

        cfg.SOLVER.BASE_LR = wandb.config.lr
    
        cfg.MODEL.LOSS_ALPHA = wandb.config.alpha

        cfg.TRAIN.BATCH_SIZE = wandb.config.batch_size
    
        cfg.SOLVER.EGONCE_HARD_NEG_MINING = wandb.config.egonce_hard_neg_mining
    
        cfg.MODEL.FACE_PARSE = wandb.config.face_parsing
        
        cfg.DATA.NUM_FRAMES = wandb.config.num_frames

        cfg.DATA.SAMPLING_RATE = wandb.config.frame_sampling_rate

        # Only do 3 epochs for grid searching hyperparameters
        cfg.SOLVER.MAX_EPOCH = 1

        # Use less Workers since wandb is also increasing the workload
        #cfg.DATA_LOADER.NUM_WORKERS = 8

    # Perform training.
    if cfg.TRAIN.ENABLE:
        if use_wandb:
            launch_job(cfg=cfg, init_method=args.init_method, func=train, wandb_run=run)
            wandb.finish()
        else:
            launch_job(cfg=cfg, init_method=args.init_method, func=train, wandb_run=None)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
   
    args = parse_args()

    use_wandb = args.use_wandb
    
    print("use_wandb : ", use_wandb)

    if use_wandb:
        sweep_config = {
            "method": "grid",
            "parameters": {
                "lr": {
                    "values": [0.0001]
                },
                "alpha": {
                    "values": [0.05]
                },
                "batch_size": {
                    "values": [2]
                },
                "face_parsing": {
                    "values": [False]
                },
                "egonce_hard_neg_mining": {
                    "values": [False]
                },
                "num_frames": {
                    "values": [2]
                },
                "frame_sampling_rate": {
                    "values": [10]
                },
            },
            "metric": {
                "name": "Train/loss",
                "goal": "minimize"
            }
        }


        sweep_id = wandb.sweep(sweep=sweep_config, project="avgaze")

        wandb.agent(sweep_id, function=main)
    
    else:
        main()
