export POLICY_CHECKPOINT_PATH=/home/nlab/taniguchi/airoa/airoa-evaluation-ICRA/src/my_policy/my_checkpoint
export POLICY_PYTORCH_DEVICE=cuda

# 1. ビルド & 起動 (TEST_MODE=true デフォルト、実機不要)
./RUN-DOCKER-CONTAINER.sh up

# 2. サーバがモデルをロードできたか確認
./RUN-DOCKER-CONTAINER.sh logs policy_server
# 期待: "server listening on 0.0.0.0:8000"

# 3. 合成観測を流す
./RUN-DOCKER-CONTAINER.sh shell
# コンテナ内:
roslaunch hsr_policy_client hsr_policy_client.launch
# policy_server のログに "Action executed." が (繰り返し) 出ることを期待

# 4. 停止
./RUN-DOCKER-CONTAINER.sh down