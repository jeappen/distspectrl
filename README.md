# DistSPECTRL: Distributing Specifications in Multi-Agent Reinforcement Learning Systems

Sample code to run DistSPECTRL.

## Installation

Create new python 3.9 environment with required packages using 

        conda create --name distspectrl python=3.9
        conda activate distspectrl
        pip install -r requirements.txt

## Project Structure

```
├── README.md
├── requirements.txt                    # requirements file installed via pip
├── train.py                            # training script
├── test_multi_exp_PPO.sh               # bash script to run multiple experiments at once
├── gym_waysharing                      # (dir) 2D and 3D Navigation Environment
├── rm_cooperative_marl                 # (dir) MARL reward machine code from Neary et al. with modifications for Ray
├── spectrl                             # (dir) spectrl code provided by Jothimurugan et al. (as-is) 
├── img                                 # (dir) sample images of results, created by functions in src.visualize
└── src
    ├── main  
    │   ├── MA_monitor                  # Class for representing composite task monitors
    │   ├── MA_spec_compiler            # Composition and compilation functions
    │   └── reward_shape_PPO_wrapper    # Specs and Wrapped Environments
    ├── env_wrappers  
    │   ├── MA_learning                 # Wrapper functions for RLLib MA Environments and DistSPECTRL
    │   └── MA_learning_logNscaling     # Wrapper functions for MA-Dec Scaling
    ├── models  
    │   └── centralized_critic          # Centralized Critic functions for CPPO mode
    ├── tests                           # (dir) simple testing scripts using unittest
    └── visualize                       # (dir) functions to plot results
```

## Usage Example


See src.main.reward_shape_PPO_wrapper.py for more example specs. From within the main directory run

    conda activate distspectrl
    # \phi_3
    python train.py --spec_id 3 --algorithm CPPO --env navenv_inlineDS --num_workers_per_device 2  --num_cpus 8 --exp_name test --horizon 400 --train_batch_size 4800
    # \phi_a
    python train.py --spec_id 0 --algorithm CPPO --env nav3D_inlineDS --num_workers_per_device 2  --num_cpus 8 --exp_name test --horizon 400 --train_batch_size 4800
    # \phi_3 MA-Dec Scaling, N=6 agents
    python train.py --spec_id 3 --algorithm CPPO --env navenv_inlineDS_logN --num_workers_per_device 2  --num_agents 6 --num_cpus 8 --exp_name test --horizon 400 --train_batch_size 4800
    # \phi_3 centralized SPECTRL, N=6 agents
    python train.py --spec_id 3 --algorithm PPO --env navenv_spectrl_sa --num_workers_per_device 2  --num_agents 6 --num_cpus 8 --exp_name test --horizon 400 --train_batch_size 4800

Training template (check train.py for more options)

    python train.py --spec_id <spec-id (int)> --algorithm <PPO|CPPO> --env <env_name> --num_workers_per_device 2  --num_agents <2-10 (int) > --num_cpus <NUM_CPUS> --exp_name <Experiment Name> --horizon <horizon> --train_batch_size <training batch size>

Environment name can be (`nav_` | `nav3D_`)  + \<env name option\>

Env. name options to choose various modes
- inlineDS - vanilla DistSPECTRL
- inlineDS_logN - MA-Dec Scaling
- inlineDS_no_monitor - no monitor state mode
- spectrl_sa - centralized SPECTRL (needs PPO algorithm)
- rmachine - our rmachine implementation based on the author's released code 

Note: Our rmachine implementation uses a modified environment in rm_cooperative_marl.src.Environments.rendezvous_continuous and is only created for one spec ($\phi_1$)

We also include a bash script `test_multi_exp_PPO.sh` to run multiple experiments at once on the same machine.

### Tests
Run `python -m unittest -v src/tests/spectrl_test.py` to test a few local specs in a centralized implementation.

## Plotting Graphs

Tools are in src/visualize
Use the notebook Plot_Graphs.ipynb 

Custom metrics of interest are `final_state_reached_mean`, `max_depth_mean` and `stage_reached_mean`.

## Acknowledgements
This code was *heavily* built off the original [SPECTRL code](https://github.com/keyshor/spectrl_tool) by Jothimurugan et al and we thank the authors for their efforts.
We also thank Neary et al for their [multi agent reward machine implementation](https://github.com/cyrusneary/rm-cooperative-marl).