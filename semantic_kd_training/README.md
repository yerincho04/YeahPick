# Semantic KD: 환경 구성부터 학습까지

이 문서는 **Linux GPU 서버의 저장소 루트**에서 실행하는 절차다. Baseline 평가는 포함하지 않는다. `KD_VQA_dataset/`의 10개 카테고리와 `scripted_action_collection/`에서 생성한 로봇 시연으로 OpenVLA LoRA를 학습한다. 경로와 학습 설정은 `semantic_kd_training/config/config.py`에서 지정한다.

## 1. 환경 구성

### 공통: OpenVLA Python 환경

```bash
cd ~/hj/YeahPick
export CONDA_ROOT="$(conda info --base)"
export CONDA_ENVS_DIR="$HOME/envs"
```

새 Conda 설치에서 Terms of Service 오류가 날 때만 다음을 실행한다.

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

```bash
bash scripts/setup/setup_openvla_env.sh
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENVS_DIR/openvla_rl4vla"
```

설치 스크립트가 Python 3.10, PyTorch 2.2.0+cu121, OpenVLA, ManiSkill, SimplerEnv 등을 설치한다. 모델 가중치는 저장소에 없으며 처음 사용할 때 다운로드된다. 교사는 `prism-dinosiglip-224px+7b`, 학생은 `gen-robot/openvla-7b-rlvla-sft_16k`다. 교사는 `NousResearch/Llama-2-7b-hf`의 설정과 토크나이저 메타데이터도 사용한다.

### VESSL Workspace 전용: 그래픽과 시스템 라이브러리

시연 수집은 SAPIEN이 GPU로 카메라 영상을 렌더링하므로, 화면이 없는 서버에서도 Vulkan을 사용한다. 기존 VESSL Workspace의 사용자 정의 이미지 `hetheiin/yeahpick-vulkan:1`은 `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`를 설정한다. 새 Workspace 생성 시 이 이미지를 선택하고, 코드와 데이터용 persistent volume은 별도로 연결한다.

**다음 apt/EGL ICD 설정은 이 VESSL 컨테이너에서 필요했던 조치다.** 일반 Linux 서버에서 NVIDIA Vulkan이 이미 동작하면 적용하지 않아도 된다.

```bash
apt-get update
apt-get install -y libgl1 libvulkan1 libcairo2 vulkan-tools
sed 's/libGLX_nvidia.so.0/libEGL_nvidia.so.0/' \
  /etc/vulkan/icd.d/nvidia_icd.json > /tmp/nvidia_egl_icd.json
export VK_ICD_FILENAMES=/tmp/nvidia_egl_icd.json
vulkaninfo --summary
```

`libgl1`은 OpenCV의 `libGL.so.1`, `libcairo2`는 공개 표지판 SVG를 변환하는 CairoSVG에 필요하다. `/tmp/nvidia_egl_icd.json`과 `VK_ICD_FILENAMES`는 새 컨테이너/터미널에서 다시 설정해야 한다. VESSL 이미지에 그래픽 드라이버가 노출되지 않으면 Python 패키지만 설치해서는 SAPIEN 렌더링이 되지 않는다.

### VESSL Workspace 전용: NCCL 파일 버전

이 Workspace에서는 `nvidia-nccl-cu12` 패키지 정보가 `2.19.3`인데 실제 `libnccl.so.2` 내용은 `2.30.7`이었던 적이 있다. 두 GPU 통신이 실패했으므로, **Python 환경 설치 후** 파일을 강제로 다시 설치한다.

```bash
python3 -m pip install --force-reinstall --no-deps --no-cache-dir \
  --index-url https://pypi.org/simple nvidia-nccl-cu12==2.19.3
```

아래 명령의 마지막 값이 `21903`이어야 한다. 패키지 버전 정보만 확인하면 파일 내용이 잘못된 경우를 발견하지 못한다.

```bash
python3 - <<'PY'
import ctypes
from importlib.metadata import distribution
p = distribution("nvidia-nccl-cu12").locate_file("nvidia/nccl/lib/libnccl.so.2")
v = ctypes.c_int()
ctypes.CDLL(str(p)).ncclGetVersion(ctypes.byref(v))
print(p, v.value)
PY
```

두 GPU의 NCCL 통신 확인은 `bash semantic_kd_training/scripts/run.sh probe`로 한다. 두 rank 모두 `all_reduce=3.0`이어야 한다. GPU 4개로 아래 설정을 바꾼 경우에는 네 rank 모두 `all_reduce=10.0`이어야 한다. 학습 스크립트는 NCCL 2.19.3 라이브러리를 우선 로드한다.

## 2. Scripted action 데이터 생성

`scripted_action_collection/input/`에는 10개 카테고리의 이미지·`vqa.jsonl`과 배경 이미지가 이미 들어 있다. 규칙 기반 컨트롤러로 큐브를 정답 타일에 놓는 시연을 수집한다. 위 VESSL EGL 설정이 필요한 서버라면 **수집 명령을 실행하는 터미널에서** `VK_ICD_FILENAMES`를 설정한다.

```bash
bash scripted_action_collection/scripts/collect.sh
```

기본값은 10개 카테고리 × 100쌍 × 원본/좌우 교환, 총 2,000개 시도다. 시간이 제한된 Workspace에서는 카테고리별로 순서대로 실행할 수 있다.

```bash
bash scripted_action_collection/scripts/collect.sh --category counting
bash scripted_action_collection/scripts/collect.sh --category public_info
```

카테고리: `counting`, `symmetry`, `celebrity`, `living_world`, `traffic`, `state`, `attribute`, `emotion`, `public_info`, `time_clock`. 재실행하면 성공·실패를 포함해 이미 기록된 에피소드는 건너뛴다. 실패한 시도를 다시 수집하려면 `--retry-failed`를 붙인다. `--count 5`는 일부 pair만 실행한다. 여러 수집 프로세스가 하나의 `output/episodes.jsonl`에 동시에 쓰지 않도록 한다.

성공 시연은 `scripted_action_collection/output/episodes/<category>/<pair_id>/<layout>/seed_<n>/`에 프레임(`frames/`), 행동(`actions.npy`), 로봇 자세(`robot_qpos.npy`)로 저장된다. `output/episodes.jsonl`에는 성공과 실패가 모두 기록되며, KD 준비 단계는 성공 기록만 사용한다. 프레임은 각 행동 **직전** 카메라 영상이다.

## 3. VQA와 action 데이터를 KD 입력으로 복사

수집이 끝난 뒤 저장소 루트에서 실행한다. `KD_VQA_dataset/`은 10개 카테고리별 100 image pair, pair당 좌우 focus 2행과 행당 VQA 5문항을 포함한다.

```bash
mkdir -p semantic_kd_training/input/dataset semantic_kd_training/input/action_data
cp -a KD_VQA_dataset/. semantic_kd_training/input/dataset/
cp -a scripted_action_collection/output/. semantic_kd_training/input/action_data/
```

필요한 구조:

```text
semantic_kd_training/input/dataset/<category>/vqa.jsonl
semantic_kd_training/input/dataset/<category>/images/...
semantic_kd_training/input/action_data/episodes.jsonl
semantic_kd_training/input/action_data/episodes/<category>/<pair_id>/<layout>/seed_<n>/...
```

다른 위치를 쓰려면 `semantic_kd_training/config/config.py`의 `dataset_root`, `action_data_root`를 수정한다.

## 4. KD 학습과 평가

두 GPU가 보이는 서버에서 다음 순서로 실행한다. `run.sh`는 Conda 환경을 활성화한다. 기본 GPU는 `0,1`이며 바꾸려면 `KD_GPUS=2,3`처럼 지정한다.

```bash
bash semantic_kd_training/scripts/run.sh prepare
bash semantic_kd_training/scripts/run.sh cache
bash semantic_kd_training/scripts/run.sh train
bash semantic_kd_training/scripts/run.sh test
```

네 단계를 연속 실행하려면 `bash semantic_kd_training/scripts/run.sh all`을 사용한다. `prepare`는 **pair 단위로 카테고리별 80/10/10 train/validation/test**를 나누고 합성 이미지, 세 종류의 manifest와 행동 정규화 통계를 `work/`에 만든다. 같은 pair의 좌우 focus·레이아웃·seed는 같은 분할에 남는다. `cache`는 VQA와 첫 로봇 화면 grounding 질문에 대해 교사 답과 feature를 한 번씩 계산해 저장한다. 중단 후 다시 실행하면 일치하는 캐시는 재사용한다.

`train`은 OpenVLA의 기존 토큰 출력 경로를 공유하는 BF16 LoRA 학습이다. 별도 VQA/action head는 아직 없다. 매 업데이트에서 action 8개와 보조 질문 8개(VQA 평균 6개, 로봇 화면 grounding 평균 2개)를 사용한다. 두 GPU에 모델 복제본을 하나씩 두고 NCCL DDP로 기울기를 합친다. 기본 3 epoch, 학습률 `1e-4`; 손실 가중치 등은 `config/config.py`에 있다.

### GPU 수 늘리기

`student_gpu_count`만 바꿔서는 GPU 사용 수가 늘어나지 않는다. 예를 들어 GPU 4개를 쓰려면 다음 세 곳을 **같은 수로** 맞춘다.

1. `semantic_kd_training/config/config.py`: `student_gpu_count: int = 4`.
2. `semantic_kd_training/scripts/run.sh`: `probe`, `train`, `all`에 있는 `torchrun --nproc-per-node=2`를 각각 `--nproc-per-node=4`로 변경.
3. 실행할 때 보이는 GPU 4개 지정: `KD_GPUS=0,1,2,3 bash semantic_kd_training/scripts/run.sh train`.

`gradient_accumulation_steps`는 GPU 수로 나누어떨어져야 한다. 기본값 8은 GPU 4개에도 그대로 사용할 수 있으며, 이 경우 **전체 업데이트당 샘플 수는 그대로 8개**, GPU당 micro-step은 4개에서 2개로 줄어든다. GPU마다 OpenVLA 전체 모델 복제본이 필요하므로 각 GPU의 메모리도 충분해야 한다. GPU 수를 늘린다고 학습 시간이 반드시 비례해서 줄지는 않는다.

결과는 `semantic_kd_training/output_ddp_bf16/`의 `metrics.jsonl`, `best_adapter/`, `latest_adapter/`에 저장된다. Validation은 매 epoch 끝에 균형 잡힌 일부 샘플에 대한 손실로 `best_adapter`를 고른다. 현재 저장 파일에는 optimizer 상태가 없어 Workspace가 종료되면 해당 지점부터 학습을 정확히 재개할 수 없다. tmux는 터미널 연결이 끊기는 경우에만 학습을 유지한다.

`test`는 기본적으로 `best_adapter`를 읽어 **전체 test VQA를 자유 생성으로 채점**하고, test pair의 원본·좌우 교환 장면에서 **실제 로봇 rollout**을 수행한다. 10개 카테고리의 100개 test pair라면 VQA 1,000문항, 로봇 200에피소드다. `test_metrics.json`에 전체·카테고리별 VQA 정답률, 질문 유형별 정답률, 로봇 성공률과 정답 타일로 먼저 접근한 비율을 기록한다. VQA 완료 결과는 `test_vqa_metrics.json`, 개별 결과는 `test_vqa_predictions.jsonl`, `test_rollout_episodes.jsonl`에 기록한다. 로봇 평가용 타일은 학습 입력 `input/dataset/`에서 `work/test_assets/`에 생성한다. GPU 두 개면 0번은 시뮬레이션, 1번은 OpenVLA에 쓴다. VESSL에서는 앞서 설정한 NVIDIA Vulkan/EGL 드라이버가 필요하다.

```bash
bash semantic_kd_training/scripts/run.sh test
# 한 번에 생성하는 시뮬레이터 수를 줄여야 할 때:
bash semantic_kd_training/scripts/run.sh test --rollout-batch-size 1
```

학습 전 모델을 같은 test pair와 질문으로 비교하려면 아래 명령을 실행한다. 기존 KD 결과는 덮어쓰지 않고 `output_ddp_bf16/base_test/`에 저장한다. 기본 모델에는 LoRA를 붙이지 않으며, 로봇 행동 토큰은 원래 체크포인트의 `sft` 정규화 통계로 변환한다. KD 모델은 학습 데이터에서 계산한 `kd_act2answer` 통계를 사용한다.

```bash
bash semantic_kd_training/scripts/run.sh test --model base
```

### 로봇 화면 grounding 테스트

학습 때 사용한 것과 같은 방식으로, 성공한 scripted episode의 **첫 로봇 카메라 프레임**과 Knowledge 지시를 주고 정답 타일을 `Left` 또는 `Right`로 직접 생성하게 한다. 기존 `evaluate.py`의 `grounding_teacher_forced_exact`는 정답 토큰을 입력한 상태의 지표이므로 자유 생성 정답률과 다르다.

기존 VQA·rollout 결과를 다시 계산하지 않고 grounding만 평가하려면:

```bash
bash semantic_kd_training/scripts/run.sh test --grounding-only
bash semantic_kd_training/scripts/run.sh test --model base --grounding-only
```

결과는 KD 모델의 경우 `output_ddp_bf16/test_grounding_metrics.json`과 `test_grounding_predictions.jsonl`에, base 모델의 경우 `output_ddp_bf16/base_test/` 아래 같은 파일명으로 저장된다. 요약에는 전체·카테고리별·원본/좌우 교환별 정답률이, 개별 기록에는 pair ID와 예상/생성 답이 들어간다. 일반 `test` 실행에도 이 평가가 포함된다.
