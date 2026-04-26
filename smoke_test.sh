export POLICY_CHECKPOINT_PATH=/home/canzone/Desktop/work/airoa/airoa-evaluation-ICRA/src/my_policy/my_checkpoint
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell           # open client container shell
# Inside the container:
roslaunch hsr_policy_client hsr_policy_client.launch