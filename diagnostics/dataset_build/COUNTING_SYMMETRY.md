# Counting / Symmetry synthetic pilot v1

직접 제작한 파일럿 데이터입니다. 공식 Act2Answer 문항이나 원본 벤치마크의
재현 데이터가 아니며, 10문항의 결과를 해당 카테고리 전체 성능으로 일반화하지 않습니다.

## 구성

각 에셋 폴더는 `ManiSkill/mani_skill/assets/carrot/` 아래에 있습니다.

| 에셋 | 문항 | 원본 PNG | GLB 타일 |
|---|---:|---:|---:|
| `counting_pilot_v1` | 10 | 20 | 20 |
| `symmetry_pilot_v1` | 10 | 20 | 20 |

- `images/`: 모델 입력에 사용할 512×512 PNG. 이미지 자체에는 번호/정답/문자가 없습니다.
- `questions.csv`: 사람이 검토할 문항표. A는 기본 배치의 왼쪽, B는 오른쪽입니다.
- `pairs.json`, `model_db.json`, `shapes/`: 기존 평가 코드가 읽는 파일입니다.
- `geometry.json`: 도형 중심/반지름, 실제 개수, 대칭 여부, PNG 해시입니다.
- `contact_sheet.png`: 검토용 요약 이미지. 정답이 표시되므로 모델 입력에 사용하지 않습니다.

## 문항 설계와 한계

Counting: 같은 색과 크기의 원 2~7개를 사용하며 각 선택지의 개수 차이는 1입니다.
정답 개수를 지시문에 명시합니다. 적은 쪽/많은 쪽 정답이 각각 5개이고,
기본 좌/우 정답도 각각 5개입니다. 배치는 고정 난수 시드로 생성합니다.
면적과 밀도가 개수와 함께 변하므로 이 파일럿만으로 순수한 수 세기 능력을
단정하지 않습니다. 다양한 물체/배경/개수 간격에 대한 일반화도 측정하지 않습니다.

Symmetry: 이미지 중앙 수직선(x=255.5)을 기준으로 한 좌우 반사 대칭을 묻습니다.
한 쌍의 원 개수, 색, 크기는 같고 비대칭 선택지는 원 하나를 수평으로 42px 이동합니다.
정답은 항상 대칭 패턴이며 좌/우 정답이 각각 5개입니다. 회전 대칭이나 다른 축의
대칭은 평가하지 않습니다. 최종 카메라 시점에서 패턴이 충분히 보이는지는 영상으로
확인해야 합니다. 3D 장면의 원근 왜곡과 타일 방향이 이미지 판단에 영향을 줄 수 있습니다.

## 조건

기본 실행은 `knowledge`와 `explicit_spatial` 두 조건입니다.
`semantic_answer`는 기존 스키마 호환을 위해 들어 있지만 `explicit_object`는
개수/대칭을 다시 묻는 표현이 되어 독립적인 grounding 대조군이 아닙니다.
따라서 기본 실험에서 제외합니다. 각 조건은 noswap과 swap을 자동 실행합니다.
총 20문항 × 2조건 × 2배치 = 80에피소드(seed 1개)입니다.

## 생성 및 검증

```bash
python diagnostics/dataset_build/build_counting_symmetry.py
```

Pillow, NumPy, SciPy, trimesh가 필요합니다. 시드 `20261005`를 사용합니다.
생성 중 연결요소로 실제 원 개수, 도형 간격/경계, 픽셀 단위 대칭, 정답 균형,
내보낸 GLB의 텍스처와 원본 PNG 일치를 검사합니다. 검증용 정답 정보는
파일 메타데이터와 요약 이미지에만 들어가며 모델 입력 그림에 넣지 않습니다.
GLB는 저장소의 `tile_asset.py`와 기존 `test_colors` 타일 템플릿을 재사용합니다.

## 실행

VESSL 원격 터미널에서 환경과 Vulkan 설정을 먼저 활성화합니다.

```bash
source /root/activate-openvla.sh
EVAL_GPU=0 SEED=0 bash scripts/eval_counting_symmetry.sh
```

카테고리 하나만 실행할 수도 있습니다.

```bash
ASSET=counting_pilot_v1 EVAL_GPU=0 bash scripts/eval_counting_symmetry.sh
ASSET=symmetry_pilot_v1 EVAL_GPU=1 bash scripts/eval_counting_symmetry.sh
```

처음에는 문항 하나를 먼저 확인합니다.

```bash
ASSETS=counting_pilot_v1 START_ID=0 COUNT=1 BUFFER_INFERBATCH=1 EVAL_GPU=0 \
  ENABLE_DIAGNOSTICS=1 INSTRUCTION_CONDITION=knowledge bash scripts/eval_openvla.sh
```

완료 후 요약:

```bash
python diagnostics/condition_summary.py counting_pilot_v1
python diagnostics/condition_summary.py symmetry_pilot_v1
```

요약의 `explicit_object`는 미실행 상태입니다. `0/0`을 실제 0% 성능으로 해석하지
마세요. 동일 seed/조건/문항을 재실행하면 기존 출력과 로그가 겹칠 수 있으므로,
새 실행은 별도 `A2A_OUTPUT_DIR`와 `A2A_LOG_DIR`를 지정하고 요약에도 해당 출력 경로를
두 번째 인자로 전달하세요. 모델 가중치와 실행 결과는 데이터셋에 포함하지 않습니다.
