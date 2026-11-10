# Celebrity · Living world · Traffic VQA dataset

사용자가 제공한 `symmetry_vqa.jsonl`과 같은 JSONL 구조로 만든 세 category의 데이터입니다.

## 구성

| Category | 개별 이미지 | 서로 다른 이미지 pair | VQA 묶음 | 개별 QA |
|---|---:|---:|---:|---:|
| celebrity | 30 | 100 | 200 | 1,000 |
| living_world | 30 | 100 | 200 | 1,000 |
| traffic | 30 | 100 | 200 | 1,000 |
| 합계 | 90 | 300 | 600 | 3,000 |

한 pair에는 left와 right를 각각 focus로 삼는 묶음이 하나씩 있습니다. 각 묶음은 다음 다섯 질문으로 구성됩니다.

1. Object recognition: focus 이미지에 묘사된 대상의 이름.
2. Attribute recognition: focus 이미지에서 보이는 특징.
3. Semantic reasoning: 관련 지식이나 의미를 만족하는 대상의 이름.
4. Visual grounding: 같은 지식이나 의미를 만족하는 대상의 Left/Right 위치.
5. Counterfactual grounding: 두 타일을 교환하거나, 위치 변경 없이 테두리를 추가한 경우의 위치.

## 질문 원본에 관한 중요 사항

첨부 자료에는 symmetry/counting 예시만 있었으며, 요청한 세 category의 **기존 200개 질문 원본은 제공되지 않았습니다**. 따라서 이번 결과는 원래 질문을 추출한 자료가 아닙니다. category마다 **새 source instruction 200개를 작성하고 그중 100개를 선택**했습니다. 해당 원본 풀과 선택 여부를 각 category의 `*_source_questions_200.jsonl`에 함께 넣었습니다. 모든 source의 `provenance`는 `newly_authored_for_this_delivery`입니다.

image pair의 연결 구조와 선택 source ID 목록은 symmetry 예시를 재사용했습니다. 각 category의 이미지 30개가 빠짐없이 등장하며, 이미지마다 6회 또는 7회 사용됩니다. 서로 같은 이미지끼리의 pair 및 순서만 뒤집은 중복 pair는 없습니다. 이번 데이터에서는 source의 비교 이미지를 교체하지 않아 `comparison_image_replaced`는 모두 false입니다.

## 이미지에 관한 사항

- 개별 이미지: RGB PNG, 512 × 512.
- 좌우 합성 pair 이미지: JPEG, 1024 × 512, 왼쪽 512 pixels가 left tile.
- 모든 이미지는 이미지 생성 모델로 만든 합성 이미지입니다.
- Celebrity는 실존 공인의 생성 묘사입니다. 이름 라벨은 생성 프롬프트에서 지정한 대상을 뜻하며, 실제 촬영 사진이나 독립적인 얼굴 인증 결과가 아닙니다.
- Living world는 실제 동물 종류를 묘사합니다.
- Traffic는 Great Britain 교통표지를 기준으로 한 생성 재현입니다. 속도 30 표지는 30 mph입니다. 공식 배포용 표지 도안이나 운전 안내서로 사용하기 위한 데이터가 아닙니다.
- 각 `image_manifest.json`에 이미지의 이름, 생성 프롬프트, 보이는 특징, 지식 조건, 참고 출처 URL, 해시를 기록했습니다.

## 파일

```text
vqa_all.jsonl                 전체 VQA 묶음 600개
pairs_all.jsonl               전체 pair 300개 및 합성 이미지 경로
vqa_questions_review.csv      전체 QA 3,000행, UTF-8 BOM
dataset_metadata.json         개수·출처·크기·생성 방식
validation_report.json        구조 및 경로 검사 결과
validate_dataset.py           검사 재실행 스크립트
load_example.py               간단한 로딩 예시
celebrity/
  images/                    30개 PNG
  pairs/                     100개 JPEG
  celebrity_vqa.jsonl         200개 묶음
  celebrity_pairs.jsonl       100개 pair
  celebrity_source_questions_200.jsonl
  image_manifest.json
  celebrity_contact_sheet.jpg
living_world/                 위와 같은 구성
traffic/                      위와 같은 구성
```

`images.left.path`와 `images.right.path`, `pair_image_path`는 이 README가 있는 dataset 최상위 폴더를 기준으로 하는 상대 경로입니다. 각 category의 JSONL을 사용할 때도 같은 최상위 폴더를 기준으로 이미지를 읽어야 합니다.

## JSONL 해석

- `group_id`: pair와 focus를 합친 고유 ID.
- `category`: celebrity / living_world / traffic.
- `pair_id`: 동일한 두 이미지로 구성된 pair의 ID.
- `focus_tile`, `focus_image_id`: 해당 묶음이 평가하는 타일과 이미지.
- `images`: 좌우 이미지의 ID와 상대 경로.
- `knowledge`: source instruction, source pair, 정답 위치, 질문 계열 정보.
- `knowledge.expected_answer`: pair의 knowledge instruction 정답으로, focus와 다를 수 있습니다.
- `vqa`: `type`, `question`, `expected_answer`를 가진 다섯 질문.

영어 질문과 답을 사용하며 위치 답은 `Left` 또는 `Right`입니다. 같은 pair의 두 묶음에서 `images`와 `knowledge`는 같습니다. Visual grounding의 정답은 해당 묶음의 focus입니다. Counterfactual grounding은 교환 시 반대 위치, 테두리 추가 시 원래 위치입니다.

## 사용 및 검증

Python과 Pillow가 있으면 다음과 같이 사용할 수 있습니다.

```bash
python validate_dataset.py
python load_example.py
```

학습/검증/시험 split은 지정하지 않았습니다. 같은 30개 이미지를 여러 pair에서 재사용하므로, pair나 질문 행만 무작위로 나누면 같은 이미지가 서로 다른 split에 들어갑니다. 독립적인 일반화 성능을 평가하려면 이미지 또는 대상 ID 단위로 split을 구성해야 합니다.
