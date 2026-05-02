export POLICY_CHECKPOINT_PATH=/home/canzone/Desktop/work/airoa/airoa-evaluation-ICRA/src/my_policy/my_checkpoint
export POLICY_CONFIG_NAME="pi05_hsr_task6891011_level12_v2.5_train_adaptive"
./RUN-DOCKER-CONTAINER.sh up               # builds & starts containers (uses TEST_MODE=true)
./RUN-DOCKER-CONTAINER.sh shell            # opens client container shell
roslaunch hsr_policy_client hsr_policy_client.launch    # sends synthetic observations
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh logs hsr_client
./RUN-DOCKER-CONTAINER.sh down