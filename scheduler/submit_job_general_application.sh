# [IMPORTANT]
# 1. Ensure your command in ${main_py} is correct (including uncommented command, various epoch definitions).
# 2. Ensure your loading and saving directory in ${main_py} is correct.
# In this file
# 1. Ensure your ${main_py} is correct
# 2. Ensure your ${target} and ${mode} are correct
# 3. Ensure your sub/group_list, trial_list, epoch_list etc are correct.
# 4. Ensure your file path of 'final_state' is correct.
proj_dir="/home/ftian/storage/projects/MFM_exploration"
scripts_dir="${proj_dir}/src/scripts"
conda_env=MFM_tzeng

dataset_name=PNC # ['HCPYA', 'PNC']
main_py="${scripts_dir}/main_${dataset_name}.py"

target='age_group'
# target='overall_acc_group_high'
# target='overall_acc_group_low'
# ['only1_group', 'age_group', 'overall_acc_group_high', 'overall_acc_group_low', 'group_dl_dataset', 'individual']
mode='train'
# ['train', 'validation', 'test', 'simulate_fc_fcd', 'EI', 'val_train_param', 'simulate_fc']
need_gpu=0

logpath="${proj_dir}/logs/${dataset_name}/${target}/${mode}"

mkdir -p ${logpath}

echo $dataset_name $target $mode

# For group
if [ ${target} = 'only1_group' ]; then # No group_nbr need

    if [ ${mode} = 'train' ]; then

        # trial_list=($(seq 1 1 1))
        trial_list=(4)
        seed_list=($(seq 2 1 10))
        # seed_list=(1)

        # Make up log directories for different trials
        for trial_nbr in "${trial_list[@]}"; do
            logdir="${logpath}/trial${trial_nbr}"
            mkdir -p ${logdir}

            for seed_nbr in "${seed_list[@]}"; do
                logerror="se${seed_nbr}_error.log"
                log_out="se${seed_nbr}_out.log"
                cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $trial_nbr $seed_nbr"
                $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 30:00:00 -mem 5G -name "se${seed_nbr}t${trial_nbr}_train" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"
            done
        done

    elif [ ${mode} = 'validation' ]; then

        # trial_list=($(seq 2 1 5))
        trial_list=(1)
        seed_list=($(seq 8 1 50))
        epoch_list=($(seq 0 1 99))

        for trial_nbr in "${trial_list[@]}"; do

            logdir="${logpath}/trial${trial_nbr}"
            mkdir -p ${logdir}

            for seed_nbr in "${seed_list[@]}"; do
                # final_state=/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/group_340/train/trial${trial_nbr}/seed${seed_nbr}/final_state.pth
                # if [ ! -f "${final_state}" ]; then
                #     echo "t${trial_nbr} se${seed_nbr} no final state."
                #     continue
                # fi

                for epoch in "${epoch_list[@]}"; do
                    logerror="se${seed_nbr}_e${epoch}_error.log"
                    log_out="se${seed_nbr}_e${epoch}_out.log"
                    cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} ${trial_nbr} ${seed_nbr} ${epoch}"
                    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 00:45:00 -mem 4G -name "e${epoch}se${seed_nbr}t${trial_nbr}_val" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"

                    if [ ${epoch} == '0' ]; then # For tackling conflicts of directory making
                        sleep 5
                    fi

                done
            done
        done

    elif [[ ${mode} = 'test' || ${mode} = 'simulate_fc_fcd' ]]; then

        trial_list=(1)
        # seed_list=($(seq 1 1 1))
        seed_list=($(seq 2 1 50))

        for trial_nbr in "${trial_list[@]}"; do
            logdir="${logpath}/trial${trial_nbr}"
            mkdir -p ${logdir}
            for seed_nbr in "${seed_list[@]}"; do
                logerror="se${seed_nbr}t${trial_nbr}_error.log"
                log_out="se${seed_nbr}t${trial_nbr}_out.log"
                cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $trial_nbr $seed_nbr"
                $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 02:00:00 -mem 5G -name "se${seed_nbr}t${trial_nbr}_test" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"
            done
        done

    fi

elif [[ ${target} = 'age_group' || ${target} = 'overall_acc_group_high' || ${target} = 'overall_acc_group_low' || ${target} = 'group_dl_dataset' ]]; then # Need to check final state manually.
    if [[ ${target} = 'age_group' ]]; then
        group_list=($(seq 1 1 29))
    elif [[ ${target} = 'overall_acc_group_high' || ${target} = 'overall_acc_group_low' ]]; then
        group_list=($(seq 1 1 14))
    elif [[ ${target} = 'group_dl_dataset' ]]; then
        group_list=($(seq 0 1 63))
    fi

    trial_list=(2)
    seed_list=(1)
    if [ ${mode} = 'train' ]; then

        # group_list=(1)
        # trial_list=($(seq 51 1 100))
        # trial_list=(1)
        # seed_list=(1)

        # Make up log directories for different trials
        for trial_nbr in "${trial_list[@]}"; do
            logdir="${logpath}/trial${trial_nbr}"
            mkdir -p ${logdir}

            for group_nbr in "${group_list[@]}"; do
                for seed_nbr in "${seed_list[@]}"; do
                    logerror="g${group_nbr}se${seed_nbr}_error.log"
                    log_out="g${group_nbr}se${seed_nbr}_out.log"
                    if [ ${need_gpu} = 1 ]; then
                        cmd="module load cuda/11.7; source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $target $group_nbr $trial_nbr $seed_nbr"
                        $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 10:00:00 -mem 4G -ngpus 1 -name "se${seed_nbr}g${group_nbr}t${trial_nbr}" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"
                    elif [ ${need_gpu} = 0 ]; then
                        cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $target $group_nbr $trial_nbr $seed_nbr"
                        $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 40:00:00 -mem 4G -name "se${seed_nbr}g${group_nbr}t${trial_nbr}" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"
                    fi
                done
            done
        done

    elif [ ${mode} = 'validation' ]; then
        # whole group range ($(seq 1 1 29)); epoch range (0, 49); trial (1, 1)
        # group_list=(1)
        # trial_list=($(seq 4 1 14))
        # trial_list=(1)
        # seed_list=(1)

        for group_nbr in "${group_list[@]}"; do

            logdir="${logpath}/group${group_nbr}"
            mkdir -p ${logdir}

            for trial_nbr in "${trial_list[@]}"; do
                for seed_nbr in "${seed_list[@]}"; do
                    for epoch in "${epoch_list[@]}"; do
                        logerror="t${trial_nbr}se${seed_nbr}e${epoch}_error.log"
                        log_out="t${trial_nbr}se${seed_nbr}e${epoch}_out.log"
                        cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $target $group_nbr $trial_nbr $seed_nbr"
                        $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 00:30:00 -mem 2G -name "e${epoch}se${seed_nbr}g${group_nbr}t${trial_nbr}_val" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"

                        if [ ${epoch} == '0' ]; then # For tackling conflicts of directory making
                            sleep 5
                        fi
                    done
                done
            done
        done

    elif [ ${mode} = 'test' ]; then
        # group_list=(1)
        # trial_list=(1)
        # seed_list=(1)
        for trial_nbr in "${trial_list[@]}"; do

            logdir="${logpath}/trial${trial_nbr}"
            mkdir -p ${logdir}

            for group_nbr in "${group_list[@]}"; do
                for seed_nbr in "${seed_list[@]}"; do
                    logerror="g${group_nbr}se${seed_nbr}_error.log"
                    log_out="g${group_nbr}se${seed_nbr}_out.log"
                    cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $target $group_nbr $trial_nbr $seed_nbr"
                    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 01:00:00 -mem 5G -name "se${seed_nbr}g${group_nbr}t${trial_nbr}_test_hybrid" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"
                done
            done
        done

    elif [ ${mode} = 'EI' ]; then
        # group_list=(1)
        # trial_list=($(seq 8 1 14))
        # trial_list=(1)
        # seed_list=(1)
        for trial_nbr in "${trial_list[@]}"; do

            logdir="${logpath}/trial${trial_nbr}"
            mkdir -p ${logdir}

            for group_nbr in "${group_list[@]}"; do
                for seed_nbr in "${seed_list[@]}"; do
                    logerror="g${group_nbr}se${seed_nbr}_error.log"
                    log_out="g${group_nbr}se${seed_nbr}_out.log"
                    cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $target $group_nbr $trial_nbr $seed_nbr"
                    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 01:00:00 -mem 2G -name "g${group_nbr}se${seed_nbr}t${trial_nbr}_EI_tz" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"
                done
            done
        done
    fi

elif [ ${target} = 'individual' ]; then

    if [ ${mode} = 'train' ]; then
        # sub_list=($(seq 401 1 885))
        sub_list=(402 406 413 414 416 422 424 430 433 434 437 450 455 473 474 477 485 498 505 507 508 510 522 530 532 533 547 552 558 566 567 568 570 573 575 576 580 582 584 587 591 598 601 603 612 626 631 632 634 637 639 643 646 648 649 652 657 658 662 670 674 675 679 682 690 696 697 698 699 701 705 707 712 713 718 723 724 727 733 736 737 742 750 751 753 754 755 756 760 762 764 769 772 773 775 777 778 780 788 792 793 797 798 799 803 808 809 811 812 815 817 823 828 831 832 835 840 843 848 849 850 852 855 858 861 868 869 877)
        trial_list=(4)
        seed_list=(1)

        # Make up log directories for different trials
        for trial_nbr in "${trial_list[@]}"; do
            logdir="${logpath}/trial${trial_nbr}"
            mkdir -p ${logdir}

            for sub_nbr in "${sub_list[@]}"; do
                for seed_nbr in "${seed_list[@]}"; do
                    logerror="s${sub_nbr}se${seed_nbr}_error.log"
                    log_out="s${sub_nbr}se${seed_nbr}_out.log"
                    cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $sub_nbr $trial_nbr $seed_nbr"
                    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 20:00:00 -mem 3G -name "s${sub_nbr}se${seed_nbr}t${trial_nbr}_${dataset_name}_tz" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"
                done
            done
        done

    elif [ ${mode} = 'validation' ]; then
        # whole group range ($(seq 860 1 1028)); epoch range (0, 99); trial (1, 1)
        # sub_list=($(seq 863 1 1028))
        # trial_list=($(seq 8 1 8))
        trial_list=(9)
        epoch_list=($(seq 0 1 99))
        # epoch_list=(0)

        for sub_nbr in "${sub_list[@]}"; do

            logdir="${logpath}/sub${sub_nbr}"
            mkdir -p ${logdir}

            for trial_nbr in "${trial_list[@]}"; do
                final_state=/home/ftian/storage/projects/MFM_exploration/logs/params/${dataset_name}Params/individual/train/trial${trial_nbr}/seed/sub${sub_nbr}/final_state.pth
                if [ ! -f "${final_state}" ]; then
                    echo "subject${sub_nbr} t${trial_nbr} no final state."
                    continue
                fi

                for epoch in "${epoch_list[@]}"; do
                    logerror="e${epoch}t${trial_nbr}_error.log"
                    log_out="e${epoch}t${trial_nbr}_out.log"
                    cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $sub_nbr ${trial_nbr} ${epoch}"
                    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 00:45:00 -mem 5G -name "e${epoch}t${trial_nbr}s${sub_nbr}_val" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"

                    if [ ${epoch} == '0' ]; then # For tackling conflicts of directory making
                        sleep 5
                    fi

                done
            done
        done

    elif [ ${mode} = 'test' ]; then
        # sub_list=($(seq 863 1 1028))
        # sub_list=(862)
        trial_list=(9)

        for sub_nbr in "${sub_list[@]}"; do
            for trial_nbr in "${trial_list[@]}"; do
                logerror="s${sub_nbr}t${trial_nbr}_error.log"
                log_out="s${sub_nbr}t${trial_nbr}_out.log"
                cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $sub_nbr ${trial_nbr}"
                $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 01:00:00 -mem 6G -name "s${sub_nbr}t${trial_nbr}_test" -joberr "$logpath/$logerror" -jobout "$logpath/$log_out"
            done
        done

    elif [ ${mode} = 'EI' ]; then
        sub_list=($(seq 1 1 884))
        # sub_list=(0)
        trial_list=(1)
        seed_list=(1)

        for trial_nbr in "${trial_list[@]}"; do
            logdir="${logpath}/trial${trial_nbr}"
            mkdir -p ${logdir}

            for sub_nbr in "${sub_list[@]}"; do
                for seed_nbr in "${seed_list[@]}"; do
                    logerror="s${sub_nbr}se${seed_nbr}t${trial_nbr}_error.log"
                    log_out="s${sub_nbr}se${seed_nbr}t${trial_nbr}_out.log"
                    cmd="source activate ${conda_env}; cd ${proj_dir}; python -u ${main_py} $sub_nbr $trial_nbr $seed_nbr"
                    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd" -walltime 01:00:00 -mem 5G -name "s${sub_nbr}se${seed_nbr}t${trial_nbr}_EI" -joberr "$logdir/$logerror" -jobout "$logdir/$log_out"
                done
            done
        done

    fi
fi
