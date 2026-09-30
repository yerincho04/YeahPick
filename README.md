<div align="center">

# Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.19297) [![Project-Page](https://img.shields.io/badge/Project--Page-%2300B4AB?style=for-the-badge&logo=logolol&logoColor=white&labelColor=000000)](https://tttonyalpha.github.io/act2answer/)  
[![HF Papers](https://img.shields.io/badge/Model--Dataset-%23FFD14D?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2606.19297)

</div>
</div>

<p align="center">
  <img src="figs/a2a_preview.png" width="100%" alt="Act2Answer overview and evaluation results">
</p>

**Act2Answer**는 로봇 적응(robotics adaptation) 이후에도 Vision-Language-Action(VLA) 모델이 상식과 세계 지식을 유지하는지 평가하는 체화(embodied) 평가 프로토콜입니다. 모델에게 텍스트로 답하게 하는 대신, 각 VLM 스타일의 질문을 짧은 탁상(tabletop) 에피소드로 바꿉니다. 에이전트는 자연어 지시문을 읽고, 정답이라고 생각하는 이미지 타일 위에 큐브를 올려놓는 방식으로 답합니다.

운동 제어(motor) 문제를 의도적으로 단순하게 유지하여, 긴 호흡의 제어 난이도보다는 지식이 없거나, 잊혀졌거나, 행동으로 접근할 수 없는 경우를 더 잘 드러내는 것이 목표입니다.

<!-- ## News:

- **[2026/06]** Act2Answer is released on arXiv: [2606.19297](https://arxiv.org/abs/2606.19297).
- **[2026/06]** Code and evaluation scripts for the Act2Answer benchmark suite are available in this repository. -->

## 목차

- [**✨ 개요**](#overview)
- [**📚 Act2Answer**](#act2answer)
- [**🔍 주요 발견**](#keyfindings)
- [**⚙️ 실험 재현하기**](#installation)
- [**❤️ 인용**](#citation)

<h2 id="overview">✨ 개요</h2>

Act2Answer는 기존의 VLM 벤치마크를 체화된 이지선다(binary-choice) 형식으로 변환합니다. 각 태스크는 행동에 적합한 짧은 지시문, 두 개의 시각적 답안 선택지, 그리고 시뮬레이션 내의 공통 선택 행동으로 구성됩니다. 이 벤치마크는 일상적인 체화 에이전트에게 중요한 지식 범주, 즉 사회적, 물리적, 정량적, 시간적, 규범적, 문화적, 생물학적 지식에 초점을 맞춥니다.

<p align="center">
  <img src="figs/preview_video_grid_v2.gif" width="95%" alt="Act2Answer task examples">
</p>

<h2 id="act2answer">📚 Act2Answer</h2>

Act2Answer 스위트는 **1,720개의 고유한 이지선다 질문**으로 이루어져 있으며, 원본 배치와 좌우를 바꾼(swapped) 배치를 모두 포함하면 **3,440개의 평가 에피소드**가 됩니다. 다섯 개의 출처 벤치마크에서 가져온 12개 범주를 다룹니다.

<p align="center">
  <img src="figs/a2a_construction.png" width="95%" alt="Act2Answer data curation pipeline">
</p>

<h2 id="keyfindings">🔍 주요 발견</h2>

<p align="center">
  <img src="figs/knowledge_probing.png" width="95%" alt="Layerwise probing results">
</p>

- 현재의 VLA는 **Color**(색상)나 **Shape**(모양)와 같은 단순한 지각적 구분은 대체로 잘 유지합니다.
- 더 풍부한 의미 범주는 훨씬 어렵습니다. **Emotion**, **Attribute**, **State**, **Time**, **Counting**, **Symmetry**, **Traffic**, **Public Info**, **Celebrity**, **Living World**는 많은 모델에서 여전히 우연 수준(chance) 근처에 머무는 경우가 많습니다.
- 강력한 VLM 베이스라인은 지식에 민감한 많은 범주에서 대응되는 VLA보다 대략 **20~40점** 더 높은 성능을 보이며, 이는 VLM과 VLA 사이에 상당한 격차가 있음을 시사합니다.
- 레이어별 프로빙(probing) 결과, 답과 관련된 정보는 백본의 중간 레이어에서 여전히 복원 가능한 경우가 많지만, 행동 예측에 사용되는 레이어 근처에서는 약해집니다.
- 지속적인 비전-언어(vision-language) 지도학습으로 학습된 VLA 모델은 주로 로봇 데이터로만 학습된 모델보다 지식에 민감한 태스크에서 더 좋은 성능을 보이는 경향이 있습니다.
- 다운스트림 행동 파인튜닝은 제어 성능을 향상시킬 수 있지만, 일부 지식에 민감한 행동은 오히려 더 약화시킬 수 있습니다.

<h2 id="installation">⚙️ 실험 재현하기 (OpenVLA)</h2>

이 저장소는 공식 Act2Answer 코드에 OpenVLA 진단 코드(`diagnostics/`)와 실행 스크립트(`slurm/`)를 추가한 것입니다. 공식 벤치마크 에셋은 아직 공개되지 않았으므로(프로젝트 페이지에 "Assets (coming soon)"), 테스트할 에셋 세트는 직접 준비해야 합니다(아래 4단계).

**무엇을 실험하는가:** 에셋 세트의 모든 질문을 세 가지 지시문으로 묻고(각각 타일 좌/우 배치 2가지), 조건별 성공률을 비교합니다.

| 조건 | 지시문 형태 |
|---|---|
| `knowledge` | 질문 그대로 (예: "Place the cube on the animal that is associated with Australia.") |
| `explicit_object` | 정답 이름을 직접 제시 (예: "Place the cube on the kangaroo.") |
| `explicit_spatial` | 정답 위치를 직접 제시 (예: "Place the cube on the left tile.") |

knowledge에서만 실패한다면 지식/추론 문제이고, explicit_object에서도 실패한다면 단어와 그림을 연결하는(grounding) 문제이며, 세 조건 모두 실패하면 동작 제어 문제일 가능성이 큽니다.

### 0. 요구 사항

- NVIDIA GPU(VRAM 24GB 권장, RTX 3090에서 검증)와 CUDA 12.1 호환 드라이버가 있는 Linux.
- Conda 또는 Miniconda, `git`, 그리고 디스크 공간 약 25GB 이상(conda 환경 + 모델 가중치 약 15GB).
- 인터넷 연결: OpenVLA 가중치(`gen-robot/openvla-7b-rlvla-sft_16k`)와 ManiSkill의 `bridge_v2_real2sim` 에셋은 첫 실행 때 자동으로 내려받습니다.
- Slurm 클러스터 (없어도 됩니다 — 아래 "Slurm 없이 실행하기" 참고).

### 1. 저장소 받기

```bash
git clone <your-repo-url> Act2Answer
cd Act2Answer
```

### 2. 내 서버에 맞게 고쳐야 하는 값 (하드코딩된 경로와 클러스터 설정)

아래 값들은 원래 개발 서버(`/home/yerincho04/YeahPick`, Slurm 파티션 `base_suma_rtx3090`)에 맞춰져 있어서 **그대로는 다른 서버에서 동작하지 않습니다.** `slurm/run_openvla.sbatch`에서 고치세요 (`slurm/`의 다른 sbatch 스크립트들도 같은 값을 쓰므로, 사용할 경우 똑같이 고쳐야 합니다).

| 위치 (`slurm/run_openvla.sbatch`) | 현재 값 | 바꿀 것 |
|---|---|---|
| 3행 `#SBATCH --partition` | `base_suma_rtx3090` | 내 클러스터의 GPU 파티션 이름 (`sinfo`로 확인) |
| 4행 `#SBATCH --gres` | `gpu:1` | GPU 1개를 요청하는 형식이 다르면 수정 |
| 5~7행 CPU/메모리/시간 | 8 CPU, 64G, 6시간 | 필요하면 조정 |
| 8, 9행 `#SBATCH --output/--error` | `/home/yerincho04/YeahPick/Act2Answer/logs/...` | 내 저장소의 절대 경로 (`#SBATCH` 줄에는 변수를 쓸 수 없어 절대 경로가 필요) |
| 13행 `PROJECT_ROOT` | `/home/yerincho04/YeahPick/Act2Answer` | 내 저장소 절대 경로 |
| 14행 `WORKSPACE_ROOT` | `/home/yerincho04/YeahPick` | conda 환경과 캐시를 둘 폴더 (저장소 바깥 권장) |
| 18, 20행 conda 위치 | `$WORKSPACE_ROOT/miniconda3.partial-20260916` | 내 conda 설치 경로. 일반 설치라면 18행을 `source <conda경로>/etc/profile.d/conda.sh`로, 20행을 `export CONDA_ROOT=<conda경로>`로 바꿈 |
| 26행 `module load cuda/12.1` | | 클러스터에 해당 모듈이 없으면 이름을 바꾸거나 삭제 |

경로를 한 번에 바꾸는 예 (저장소 안에서 실행):

```bash
sed -i "s#/home/yerincho04/YeahPick/Act2Answer#$PWD#g; s#/home/yerincho04/YeahPick#$(dirname $PWD)#g" slurm/run_openvla.sbatch
```

그 외 참고:

- `scripts/env.sh`의 `CONDA_ENVS_DIR` 기본값은 `/home/jovyan/.mlspace/envs`입니다. 환경변수로 덮어쓰세요 (예: `export CONDA_ENVS_DIR=$HOME/envs`). sbatch 스크립트는 이미 `$WORKSPACE_ROOT/envs`로 설정합니다.
- `SETUP_README.md`는 공식 저장소의 문서라 이 실험에 필요 없는 모델의 설치도 다룹니다. 이 저장소에는 OpenVLA 관련 스크립트만 남아 있습니다.

### 3. 환경 만들기 (OpenVLA 전용, 한 번만)

```bash
export CONDA_ENVS_DIR=<환경을 둘 경로>      # 예: $HOME/envs
export CONDA_ROOT=<conda 설치 경로>           # 예: $HOME/miniconda3
bash scripts/setup/setup_openvla_env.sh       # conda 환경 openvla_rl4vla 생성 (torch 2.2.0 + cu121, flash-attn 등)
```

### 4. 에셋 세트 준비하기

`ManiSkill/mani_skill/assets/carrot/<asset_name>/` 폴더에 에셋 세트를 둡니다 (`pairs.json`, `model_db.json`, `shapes/`). 공식 저장소에는 `test_colors`만 들어 있고, 동작 확인용으로 쓸 수 있습니다(`knowledge`, `explicit_spatial` 조건만 해당 — `explicit_object` 조건에는 아래 `semantic_answer`가 필요합니다).

`pairs.json`의 각 항목에 필요한 필드: `index`, `left`, `right`(타일 이름), `question`, `answer`("Left"/"Right"), 그리고 선택적으로 `knowledge_instruction`(없으면 `question`을 사용), `semantic_answer`(`explicit_object` 조건에 필수), `left_label`, `right_label`, `knowledge_category`. 자세한 예시는 `diagnostics/dataset_build/build_dataset.py`(에셋을 코드로 생성하는 예시 빌더)를 참고하세요.

### 5. 실행

`ASSETS`에 에셋 폴더 이름을 넣고, 질문 수에 맞춰 배열 크기를 계산해 제출합니다 (한 작업이 질문 5개 × 두 타일 배치를 처리하며, 조건은 3개):

```bash
ASSETS=<asset_name>
N=$(python3 -c "import json;print(len(json.load(open('ManiSkill/mani_skill/assets/carrot/$ASSETS/pairs.json'))))")
SHARDS=$(( (N + 4) / 5 ))
sbatch --array=0-$((3 * SHARDS - 1)) --export=ALL,ASSETS=$ASSETS,SEED=0 slurm/run_openvla.sbatch
```

진행은 `squeue -u $USER`, 로그는 `logs/slurm_openvla_*.out/.err`에서 확인합니다. 결과는 `outputs/openvla-<asset_name>-<조건>-<배치>-q..-seed0/`에 저장됩니다 (영상, `stats.yaml`, `diagnostics/episode_log.jsonl`). 한 작업에 질문을 몇 개 처리할지는 `SHARD` 환경변수로 바꿀 수 있으며(기본 5), 위 계산의 `5`/`4`도 같이 바꿔야 합니다. 24GB GPU에서는 한 번에 5개 환경이 안전합니다.

**Slurm 없이 실행하기** (GPU가 있는 서버에서 직접. `CONDA_ENVS_DIR`/`CONDA_ROOT`를 먼저 설정하세요):

```bash
for cond in knowledge explicit_object explicit_spatial; do
  for start in 0 5 10; do   # 질문 수에 맞게 5씩 증가시킴 (마지막 조각은 COUNT를 남은 개수로)
    ENABLE_DIAGNOSTICS=1 ASSETS=<asset_name> START_ID=$start COUNT=5 BUFFER_INFERBATCH=5 \
      EVAL_GPU=0 INSTRUCTION_CONDITION=$cond SEED=0 bash scripts/eval_openvla.sh
  done
done
```

### 6. 결과 보기

```bash
python3 diagnostics/condition_summary.py <asset_name>
```

조건별 성공률과 질문별 성공 여부(noswap/swap)가 출력됩니다. 성공률이 50% 근처이면서 질문과 상관없이 같은 쪽 타일만 고르는 패턴(예: 항상 왼쪽)이라면, 모델이 지시문이나 이미지를 사용하지 않고 위치 편향으로 움직이고 있다는 뜻입니다. 한 번의 seed와 적은 질문 수로는 결론을 내리기 어려우니 질문 수와 `SEED`를 늘려 확인하세요.

### 7. 자신의 GitHub 저장소에 올리기

이 저장소의 `origin`은 원래 공식 저장소(`CognitiveAISystems/Act2Answer`)를 가리킵니다. 새 저장소를 만든 뒤 `git remote set-url origin <새-저장소-URL>`로 바꾸고, 공식 저장소를 계속 참고하려면 `git remote add upstream https://github.com/CognitiveAISystems/Act2Answer.git`를 추가하세요. `diagnostics/`, `slurm/` 등 새로 만든 파일은 `git add`가 필요하고, `outputs/`와 `logs/`는 올리지 않는 것이 좋습니다.

<h2 id="citation">❤️ 인용</h2>

Act2Answer가 유용하다면 다음 논문을 인용해 주세요:

```bibtex
@misc{kachaev2026doesvlaknowbasics,
  title={Does VLA Even Know the Basics? Measuring Commonsense and World Knowledge Retention in Vision-Language-Action Models},
  author={Nikita Kachaev and Andrey Moskalenko and Matvey Skripkin and Nikita Kurlaev and Daria Pugacheva and Albina Burlova and Mikhail Kolosov and Denis Shepelev and Andrey Kuznetsov and Elena Tutubalina and Aleksandr I. Panov and Alexey K. Kovalev and Vlad Shakhuro},
  year={2026},
  eprint={2606.19297},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.19297}
}
```

## 감사의 말

Act2Answer는 [SimplerEnv](https://github.com/simpler-env/SimplerEnv)와 [ManiSkill](https://github.com/haosulab/ManiSkill)을 기반으로 하며, 평가 하네스의 일부는 [RL4VLA](https://github.com/gen-robot/RL4VLA)에서 가져왔습니다. README 구조는 공개된 [BlindVLA](https://github.com/CognitiveAISystems/BlindVLA) 프로젝트 스타일을 따릅니다.
