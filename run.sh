clear
# Streaming Sin
parallel --eta --ungroup python main.py --config_file ./configs/streamsin.json --config_idx {1} ::: $(seq 1 2940)
parallel --eta --ungroup python main.py --config_file ./configs/streamsin_edit0.json --config_idx {1} ::: $(seq 1 2)
python main.py --config_file ./configs/streamsin_edit1.json --config_idx 1
python main.py --config_file ./configs/streamsin_edit2.json --config_idx 1
# Simple RL
parallel --eta --ungroup python main.py --config_file ./configs/mc_dqn.json --config_idx {1} ::: $(seq 1 2400)
parallel --eta --ungroup python main.py --config_file ./configs/copter_dqn.json --config_idx {1} ::: $(seq 1 2400)
parallel --eta --ungroup python main.py --config_file ./configs/acrobot_dqn.json --config_idx {1} ::: $(seq 1 2400)
parallel --eta --ungroup python main.py --config_file ./configs/catcher_dqn.json --config_idx {1} ::: $(seq 1 2400)
# Mujoco
parallel --eta --ungroup python main.py --config_file ./configs/mujoco_ppo.json --config_idx {1} ::: $(seq 1 300)
parallel --eta --ungroup python main.py --config_file ./configs/mujoco_sac.json --config_idx {1} ::: $(seq 1 900)
# Atari
parallel --eta --ungroup python main.py --config_file ./configs/atari_rainbow.json --config_idx {1} ::: $(seq 1 300)
parallel --eta --ungroup python main.py --config_file ./configs/atari_dqn.json --config_idx {1} ::: $(seq 1 300)
parallel --eta --ungroup python main.py --config_file ./configs/grad_atari_dqn.json --config_idx {1} ::: $(seq 1 60)
parallel --eta --ungroup python main.py --config_file ./configs/grad_atari_rainbow.json --config_idx {1} ::: $(seq 1 60)