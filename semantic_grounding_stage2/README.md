# Semantic grounding Stage 2

Stage 1의 best adapter에서 이어서 학습한다. 첫 화면에서 Left 또는 Right를 한 번 고르고, 그 방향을 고정해 매 step의 OpenVLA action token 7개를 생성한다.

## 입력과 실행

STAGE1_RUN은 완료된 semantic_grounding_stage1/runs/<run> 경로다. 프로젝트 루트의 KD_VQA_simulation_dataset과 action_data를 input/dataset, input/action_data로 자동 링크한다. action_data에는 22 cm 타일의 episodes.jsonl 및 episodes 디렉터리가 필요하다. 다른 저장 위치라면 input 하위에 직접 링크하거나 복사하면 된다.

~~~bash
cd ~/hj/YeahPick
export STAGE1_RUN="$PWD/semantic_grounding_stage1/runs/<stage1-run>"
export TRAIN_GPUS=0,1
export MODEL_CACHE_ROOT=/opt/yeahpick-cache

bash semantic_grounding_stage2/scripts/run.sh collect-counterfactual --retry-failed
bash semantic_grounding_stage2/scripts/run.sh prepare
bash semantic_grounding_stage2/scripts/run.sh train
bash semantic_grounding_stage2/scripts/run.sh test semantic_grounding_stage2/runs/<stage2-run>
~~~

수집은 --category counting, --start-pair 0 --count 20, --layout original 등으로 나눌 수 있다. 같은 카테고리의 manifest에 여러 프로세스가 동시에 쓰지 않도록 한다. 성공한 episode는 재실행 시 건너뛴다.
학습 후 validation 중 중단됐다면 다시 학습할 필요 없이 아래 명령으로 남은 epoch 평가와 best 선택을 이어간다.

~~~bash
bash semantic_grounding_stage2/scripts/run.sh select semantic_grounding_stage2/runs/<stage2-run>
~~~

## 학습 및 평가

- 반대 방향 시연은 Stage 1 train pair에만 만든다. 여기서 success는 지시문의 정답이 아니라 지정한 반대쪽 타일에 놓았다는 뜻이다.
- prepare는 Stage 1 split을 그대로 사용하며 양쪽 layout 시연이 누락되거나 반대 방향 시연이 validation/test에 섞이면 중단한다.
- 각 update에서 정답 방향을 조건으로 한 action CE와 첫 화면 방향 CE를 학습한다. VQA loss는 사용하지 않는다. 설정은 config.py에 있다.
- train은 timestamp별 run과 epoch별 adapter를 저장한다. validation에서 카테고리당 2 pair의 원본·교환 화면을 실행하고 predicted 방향의 strict tile success로 best.json을 정한다.
- test는 test pair에서 predicted, oracle, opposite 세 모드를 기록한다. 방향은 첫 화면에서 한 번 선택하여 고정한다.

결과는 runs/<run>/validation 및 runs/<run>/test_<timestamp> 아래에 저장된다.
