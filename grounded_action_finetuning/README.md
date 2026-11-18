# Grounded action fine-tuning

KD `best_adapter`에서 시작해 **로봇 화면의 정답 타일 선택을 action에 실제로 전달**하는 후속 학습이다. 학습은 기존 OpenVLA action 토큰 경로와 LoRA 가중치를 사용한다.

## 입력

| 경로 | 내용 | 이 저장본 |
|---|---|---|
| `input/dataset/<category>/vqa.jsonl`, `images/` | 10개 카테고리 VQA 데이터 | `KD_VQA_dataset`에서 복사됨 |
| `input/action_data/episodes.jsonl`, `episodes/` | 성공한 scripted 시연과 프레임·action | 서버에서 복사 필요 |
| `input/kd_adapter/adapter_config.json`, `adapter_model.safetensors` | 이전 학습의 `best_adapter` | 서버에서 복사 필요 |

서버의 repo 루트에 원본 출력이 있으면 `run.sh train`이 두 입력을 자동으로 복사한다. 다른 위치에 있다면 아래처럼 직접 채운다. `output_ddp_bf16`의 **best_adapter**를 사용하며 `latest_adapter`를 사용하지 않는다.

```bash
cp -a scripted_action_collection/output/. grounded_action_finetuning/input/action_data/
cp -a semantic_kd_training/output_ddp_bf16/best_adapter/. grounded_action_finetuning/input/kd_adapter/
```

기존 `openvla_rl4vla` conda 환경, 두 GPU, NCCL 2.19.3을 사용한다. 환경과 GPU 수, 학습률 등은 `config.py`에서 지정한다. VESSL의 Vulkan/EGL 설정은 로봇 rollout 평가에 필요하다.

## 실행

```bash
bash grounded_action_finetuning/scripts/run.sh train
```

이 명령은 `prepare.py`로 기존과 동일한 pair 단위 80/10/10 분할과 action 정규화 통계를 만든 후 학습한다. **실행마다** `runs/YYYYMMDD_HHMMSS_ffffff_UTC/`를 새로 만든다. 이 폴더에 `metrics.jsonl`, `run_config.json`, 1,000 step마다의 checkpoint, epoch별 `checkpoints/epoch_XX/`, 가장 좋은 checkpoint를 가리키는 `best.json`, 가장 최근 checkpoint를 가리키는 `latest.json`이 남는다. Epoch checkpoint는 validation **전에** 저장한다. 이전 run은 덮어쓰지 않는다. 중간에 종료한 run은 그대로 남고 다음 실행은 새 timestamp 폴더에서 시작한다. Optimizer 상태는 저장하지 않으므로 checkpoint에서 재시작하더라도 정확히 같은 학습 상태를 복원하는 것은 아니다.

학습에서는 로봇 화면 grounding 정답 `Left`/`Right`를 직접 지도하고, 같은 방향을 action 프롬프트에 넣는다. Action은 매 micro-step 학습한다. 보조 질문은 robot grounding 75%, 기존 VQA 25%로 뽑아 VQA 능력도 유지한다. 이 단계에서는 정답률이 낮았던 교사의 robot grounding feature를 KD loss로 사용하지 않는다. Validation은 모든 validation grounding 문항과 카테고리·질문 유형별 VQA 일부 문항을 평가한다. `Left`와 `Right`가 여러 토큰이어도 각 답의 전체 토큰 확률을 비교한다. Grounding 정답률을 우선으로 checkpoint를 고르고, 같은 정답률이면 action loss가 낮은 쪽을 고른다.

학습이 끝나면 출력된 `RUN_DIR`로 평가한다.

```bash
bash grounded_action_finetuning/scripts/run.sh test grounded_action_finetuning/runs/<timestamp>
```

평가는 **predicted**(모델이 첫 로봇 화면에서 고른 방향)와 **oracle**(정답 방향) 두 rollout을 모두 수행한다. 기본은 test pair 100개 × 원본/좌우 교환 × 두 모드다. 시간이 제한되면 `--mode predicted` 또는 `--mode oracle`로 하나만 실행할 수 있다. 각 평가도 run 안에 고유 timestamp 폴더를 만들고, 모드별 `*_episodes.jsonl`과 `metrics.json`을 저장한다. `predicted`와 `oracle`의 차이가 크면 타일 선택이, 둘 다 낮으면 action 실행이 주된 병목이다.

현재 코드의 초기 조건은 scripted 시연의 첫 프레임이다. 모델은 그 화면에서 한 번 정답 방향을 고른 다음, 에피소드 내내 같은 방향을 action 입력으로 사용한다. 실제 실행 결과는 서버 GPU와 시뮬레이터에서 확인해야 한다.
