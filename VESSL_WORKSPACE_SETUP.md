# VESSL Workspace 설정 (OpenVLA / Public Info)

## 1. Container image 설정

VESSL의 **Create Workspace → Container image → Custom**에 `hetheiin/yeahpick-vulkan:1`을 입력한다. 기존 코드와 환경을 재사용할 경우 기존 영구 볼륨을 **같은 경로**에 연결한다. 임시 저장소의 파일은 새 Workspace로 자동 이전되지 않는다.

## 2. 새 Workspace에서 실행

수정된 `YeahPick` 코드와 `public_info_v1` 에셋이 새 Workspace에 있어야 한다. 특히 `scripts/eval_public_info_v1.sh`, `scripts/setup/setup_openvla_env.sh`, `SimplerEnv/simpler_env/env/simpler_wrapper_v4.py`, `requirements/openvla.txt`의 로컬 수정본을 포함한다.

```bash
cd ~/hj/YeahPick
apt-get update
apt-get install -y libgl1 libvulkan1

# NVIDIA 550 드라이버의 headless Vulkan ICD는 EGL 라이브러리로 지정한다.
sed 's/libGLX_nvidia.so.0/libEGL_nvidia.so.0/' /etc/vulkan/icd.d/nvidia_icd.json > /tmp/nvidia_egl_icd.json
export VK_ICD_FILENAMES=/tmp/nvidia_egl_icd.json

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
export CONDA_ROOT="$(conda info --base)"
export CONDA_ENVS_DIR="$HOME/envs"
bash scripts/setup/setup_openvla_env.sh

bash scripts/eval_public_info_v1.sh
```

설치 스크립트가 `openvla_rl4vla` 환경을 생성한다. 실험 스크립트는 세 조건의 10문제를 5개씩 나눠, 각 묶음을 원본 및 좌우 교환으로 실행한다. `Ctrl+C`로 전체 실행을 종료할 수 있다. 모델 가중치는 캐시가 없으면 처음 실행할 때 내려받는다.

새 터미널에서 실험할 때는 `VK_ICD_FILENAMES`를 다시 설정한다. Workspace가 재생성되어 `/tmp` 파일이 사라졌다면 위 `sed` 명령부터 다시 실행한다.
