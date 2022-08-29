#/bin/bash
num_agents=3
num_cpus=17
algo="CPPO" # or CPPO
horizon=400
t_iter=5000
batch_size=15000
#declare -a algos=("a2c" "ppo2"                                                                                                                                                                                                                                                                                                                                                                                       "trpo" "ddpg" "sac" "td3")
declare -a specs=(3)
declare -a envs=("navenv_inlineDS_logN" "navenv_inlineDS") # other options : ("navenv_inlineDS_logN" "navenv_inlineDS_no_monitor")
current_date=$(date +%s) 
CONDA_PATH="~/miniconda3/etc/profile.d/conda.sh" # change appropriately
CURRENT_FILE=$(readlink -f  $0)
CURRENT_DIR=$(dirname $CURRENT_FILE)

for env in "${envs[@]}"
do
    env_folder=${env}
    for spec in "${specs[@]}"
    do
        # unique ID for each expt
        uid=test_a${algo}_n${num_agents}_${env}_${spec}_cpu${num_cpus}_hzn${horizon}

        screen -S "multiexp1_${uid}" -dm bash -c "source ${CONDA_PATH};cd ${CURRENT_DIR} ;
        conda activate  distspectrl;
        python train.py --spec_id ${spec} --num_cpus ${num_cpus} --algorithm ${algo} --env ${env} --num_agents ${num_agents} --exp_name ${uid} --horizon ${horizon} --training_iterations ${t_iter} --num_workers_per_device 2 --train_batch_size ${batch_size}; sleep 10"


    done
done
