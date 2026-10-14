# K → G → A 가설 파일럿 (연구제안서 검증용)

연구제안서 전체를 구현하는 것이 아니라, 핵심 가설 하나를 작게 검증하는 go/no-go 파일럿이다.
**판정 기준은 실험 실행 전에 이 문서에 고정했다.** 결과를 보고 기준을 바꾸지 않는다.

## 가설

> VLA 내부에 올바른 world-knowledge answer(K)가 recoverable하더라도, 이를 visual scene의
> physical target(G)으로 grounding하는 과정에서 정보가 손실된다.

## 설계

- **모델/환경:** OpenVLA (`gen-robot/openvla-7b-rlvla-sft_16k`), SimplerEnv/ManiSkill, seed 0.
- **에셋:** `semantic_repeated_v2`. 정답 10개 × 서로 다른 질문 표현 5개 = 50문제 (×noswap/swap = 100 에피소드/조건).
  생성: `cd diagnostics/dataset_build && python build_followups.py semantic_repeated_v2`
- **v1 대비 수정:** v1은 오답 배치 때문에 *두 타일 그림만 보고도* 50문제 중 45문제의 정답을 알 수 있었다
  (각 그림 쌍이 한 가지 정답으로만 등장). 그러면 K probe가 지식 없이 그림만으로 높은 점수를 낼 수 있다.
  v2는 오답 오프셋을 {±1, ±2, 5}로 바꿔 모든 그림 쌍이 두 그림 각각을 정답으로 한 번씩 등장하게 했다
  (그림만으로 정답이 드러나는 쌍: 0/25).
- **조건 (같은 장면, 같은 초기 상태, 지시문만 다름):**

| 조건 | 지시문 | 역할 |
|---|---|---|
| `knowledge` | Place the cube on the answer to this question: Which insect produces honey? | Knowledge → Object → Location → Action |
| `tile_object` | Place the cube on the tile showing the bee. | Object → Location → Action |
| `explicit_spatial` | Place the cube on the left tile. | Location → Action (motor floor) |
| `neutral` | Place the cube on one of the tiles. | 정보 없는 대조군: 장면만으로 probe가 얻는 정보량 |

- **Hidden state:** `knowledge`, `neutral` 조건에서 첫 추론 시점(t=0)의 마지막 prompt 토큰, LLM 깊이 25/50/75/100%.
- **Probe (5-fold, 각 fold에 정답마다 처음 보는 질문 표현 1개, noswap/swap은 같은 fold):**
  - **K2**: 정답 identity probe를, 장면에 있는 두 타일 중 어느 것이 정답인지로 평가 (chance 50%).
    v2 설계상 장면만으로는 50%를 넘을 수 없으므로, `neutral` 대비 상승분이 곧 지시문/지식에서 온 K 정보다.
  - K10: 10개 정답 중 identity (chance 10%, 참고용).
  - **G**: 정답 위치 left/right (chance 50%).
- **행동:** placement success, intent(실제 팔 이동 방향이 정답 타일을 향했는지), 왼쪽 선택 비율.

## 판정 기준 (실행 전 고정)

레이어 4개를 보므로 유의수준은 Bonferroni 보정 p < 0.0125.

**가설 지지 (GO)** — 다음을 모두 만족하는 레이어가 하나 이상 있을 때:
1. **K 존재:** K2(knowledge) ≥ 65% 이고, 같은 장면의 K2(neutral)보다 높음 (paired McNemar p < 0.0125).
2. **G 손실:** 같은 레이어에서 G(knowledge)의 95% 구간이 50%를 포함 (chance와 구분 안 됨).
3. **행동 일치:** knowledge의 intent 정확도 95% 구간이 50%를 포함하고, explicit_spatial의 intent 정확도 ≥ 80%.

**그 외 결과의 해석:**
- K2(knowledge)가 neutral보다 높지 않음 → 이 측정으로는 K가 표현에 없음 (F1 Knowledge Retrieval 쪽이거나 probe 한계). 가설 미지지.
- K와 G가 모두 decode되는데 intent가 틀림 → 병목은 G → A (F3 Action Generation). 가설 미지지, 다른 병목.
- explicit_spatial intent < 80% → motor floor가 낮아 파일럿 판정 불가 (inconclusive).
- tile_object도 chance 수준 → 지식이 없어도 그림-위치 grounding 자체가 실패. 기준 2와 같은 방향의 증거로 보고, 따로 기록.

## 한계 (결과 해석 시 명시)

- 공식 Act2Answer 에셋이 아닌 직접 그린 그림 타일 (공식 에셋 미공개). 정지 표지판 그림에는 "STOP" 글자가 있음.
- 정답 10개, seed 1개, 조건당 100 에피소드 — 파일럿 규모.
- Probe는 "decodable"을 의미할 뿐, 모델이 그 정보를 실제로 *사용*하는지는 보여주지 않음 (그건 causal tracing 단계).
- t=0 단일 시점, 선형 probe만 사용.

## 실행

```bash
cd diagnostics/dataset_build && python build_followups.py semantic_repeated_v2 && cd ../..
sbatch --array=0-39 --export=ALL,ASSETS=semantic_repeated_v2,SEED=0,CONDITIONS="knowledge tile_object explicit_spatial neutral" slurm/run_openvla.sbatch
python3 -m diagnostics.hypothesis_pilot --asset semantic_repeated_v2 --out-dir outputs/hypothesis_pilot
```

(질문 50개 ÷ 작업당 5개 = 10 조각 × 조건 4개 = 배열 0–39.)

## 결과 (2026-10-01, Slurm job 2365769, 400 에피소드, 오류 0)

전체 표: `outputs/hypothesis_pilot/report.md` (사전 등록 분석), `outputs/hypothesis_pilot_imagepair/report.md` (사후 robustness check).

**판정: GO — 사전 등록한 세 기준을 모두 만족.**

| 기준 | 결과 (25% 레이어) | 충족 |
|---|---|---|
| 1. K 존재 | K2(knowledge) 87% [79–92], neutral 16%, paired p < 0.0001 | ✓ |
| 2. G 손실 | G(knowledge) 50% [40–60] | ✓ |
| 3. 행동 일치 | knowledge intent 52% [42–62], explicit_spatial intent 88% | ✓ |

행동 (조건당 100 에피소드):

| 조건 | intent 정확도 | 양쪽 배치 모두 정답 (chance 25%) | 왼쪽 선택 |
|---|---|---|---|
| knowledge | 52% | 20% | 73% |
| tile_object | 49% | 16% | 75% |
| explicit_spatial | 88% | 78% | 43% |
| neutral | 57% | 32% | 35% |

- knowledge ≈ tile_object (McNemar p = 0.76): 정답 이름을 직접 알려줘도 나아지지 않음 → 실패는 지식 검색이 아니라 그림→위치 grounding 단계.
- explicit_spatial은 잘 됨 → motor floor 문제 아님.
- P(G wrong | K correct) = 51% (25% 레이어). K는 깊이에 따라 약해짐 (K2 87% → 58% at final).

**사후 robustness check (사전 등록 아님):** 사전 등록 분석에서 neutral이 chance보다 훨씬 낮게 나왔다 (K2 16%, G 18–25%).
대칭 설계에서 질문 표현 단위로 fold를 나누면, 테스트 질문의 같은 그림 배치가 학습 데이터에 *반대* 정답으로 들어가 있어
장면을 외우는 probe가 반대로 맞히게 되기 때문이다. 이 효과가 G(knowledge)의 실제 신호를 상쇄해 50%처럼 보이게 할 수 있으므로,
같은 그림 쌍의 질문을 같은 fold에 묶어 다시 분석했다 (`--fold-by image_pair`).
→ neutral이 chance로 돌아오고 (K2 46–49%, G 49–53%), 결론은 유지됨: K2(knowledge) 80% [71–87] vs neutral 47% (p < 0.0001),
G(knowledge) 49% [39–59] (25% 레이어). 최종 레이어에서 G가 60% [50–69]로 약간 오르지만 neutral과 유의한 차이는 없음 (p = 0.16).

**해석상 주의:** probe 결과는 "decodable"이지 "사용됨"이 아니다. K가 질문 속 범주 단어(예: insect → bee)에서 일부 올 수 있다.
직접 그린 그림 타일(SigLIP 인식률 10/10), seed 1개, 정답 10개의 파일럿 규모다. 다음 단계는 연구제안서대로 causal tracing
(K가 decode되는 레이어에서 activation patching으로 G/intent가 바뀌는지)과, 공식/실사 에셋·여러 seed로의 재현이다.

---

# 실사(real photo) 재현: State / Attribute / Emotion (실행 전 고정, 2026-10-02)

같은 가설과 같은 판정 기준을, 직접 그린 그림 대신 **실제 사진**으로 카테고리별로 다시 검증한다.

- **에셋:** `photo_state_v1`, `photo_attribute_v1`, `photo_emotion_v1`
  (`diagnostics/dataset_build/build_vaw_photo_sets.py`). 사진은 Visual Genome (CC BY 4.0), 라벨은 VAW
  (Adobe Research License, 비상업 연구용). 사진별 출처는 `photo_output/<asset>_credits.csv`.
- **구성:** 카테고리마다 같은 물체의 반대 상태 쌍 (State: open/closed door, wet/dry road, clean/dirty toilet, empty/full plate;
  Attribute: wooden/metal bench, glass/plastic bottle, leather/fabric chair; Emotion: happy/sad, angry/calm, surprised/sleepy person).
  라벨마다 라벨 단어가 없는 지식 질문 5개, 질문마다 서로 다른 사진. 라벨 A의 j번째 질문은 사진 (A_j, B_j),
  라벨 B의 j번째 질문은 (B_j, A_j) → 사진만으로 정답이 드러나는 쌍 0개.
- **사진 선별:** VAW에서 해당 라벨이 있고 반대 라벨이 없는 인스턴스를 잘라낸 뒤, SigLIP이 두 라벨 중 맞게 고르고
  확률 ≥ 0.8인 사진만, 라벨당 상위 5장. 5장을 못 채운 쌍은 제외 (제외 여부는 빌드 로그와 audit에 기록).
- **조건/분석:** pilot과 동일 (knowledge, tile_object, explicit_spatial, neutral; seed 0; `hypothesis_pilot.py`).
  K2 chance 50%, K10 대신 K-전체(라벨 수만큼의 클래스). robustness check의 image-pair fold는 같은 두 사진을 쓰는
  질문끼리 묶는다 (`image_key`).
- **판정:** 카테고리마다 위의 GO 기준 1–3을 그대로, 각각 독립적으로 적용한다 (Bonferroni p < 0.0125, 레이어 4개).
  세 카테고리의 결과를 합치지 않고 따로 보고한다.
- **추가 한계:** 장면 사진을 잘라낸 것이라 배경이 섞임; 사람 사진의 표정은 작을 수 있음 (SigLIP 필터로 완화);
  질문은 Claude가 작성한 초안으로 팀 검토 전임; 라벨 쌍이 3–4개뿐이라 K-전체 probe는 클래스가 적음.

**사진 선별 수정 (결과를 보기 전, 2026-10-02):** 첫 빌드의 contact sheet 육안 검토에서 잘못된 사진이 발견되어
평가 작업을 취소하고 선별을 강화했다. (1) **물체 확인(object gate):** SigLIP이 해당 물체(door, plate, person 등)를
비슷한 물체(냉장고, 쟁반, 고양이·개 등)보다 높게 골라야 통과 — 첫 빌드에서 "sad person"에 개·고양이, "open door"에
열린 냉장고가 섞였음. VAW의 "face" 객체(동물 얼굴 포함)는 제외. (2) **육안 검토 제외:** 물체/상태가 불분명한 사진 6장을
VAW instance id로 제외 (`REJECTED_INSTANCES`). (3) **Emotion:** 장면 사진 속 얼굴이 작아 후보를 넓힘 (최소 크기 60px,
후보 300장, 상태 확률 ≥ 0.6; 물체 확인은 동일). 제외된 쌍(예: wet/dry road)은 빌드 로그와 audit에 기록. 판정 기준은 변경 없음.

## 실사 결과 (2026-10-02, eval jobs 2369882–2369884, 480 에피소드, 오류 0)

전체 표: `outputs/photo_hypothesis/<asset>/<folds>/report.md`, 요약: `outputs/photo_hypothesis/SUMMARY.md`.

| 에셋 | 질문 수 | 판정 (사전 등록) | 판정 (robustness) | knowledge intent | tile_object | spatial | K2 (robust fold) |
|---|---|---|---|---|---|---|---|
| photo_state_v1 (door, toilet, plate) | 30 | **NO-GO** | NO-GO | 50% | 50% | 80% | 32–43% (chance 50%) |
| photo_attribute_v1 (bench, bottle) | 20 | **INCONCLUSIVE** (spatial 75%) | INCONCLUSIVE | 55% | 48% | 75% | 35–58% |
| photo_emotion_v1 (surprised/sleepy) | 10 | GO | **NO-GO** | 45% | 55% | 80% | 60–75% (최고 15/20, chance 대비 p = 0.04) |

- **행동은 그림 파일럿과 같다:** knowledge ≈ tile_object ≈ chance, explicit_spatial은 높음 → 정답 이름을 알려줘도 소용없고, motor 문제가 아님.
- **다른 점은 K:** 실사에서는 State/Attribute 모두 K가 hidden state에서 decode되지 않음 (chance 이하/근처). 즉 "K는 있는데 G에서 손실"이 아니라,
  K 자체가 (적어도 선형 probe로는) 보이지 않음 → 사진의 상태/재질을 지식 질문과 연결하는 단계부터 실패하거나 probe가 못 잡는 것.
- **Emotion GO는 robust하지 않음:** 사전 등록 분석의 GO는 neutral이 chance 아래로 밀리는 fold artifact에 기댄 것. image-pair fold에서는 K2 75%가
  neutral 50%와 유의하게 다르지 않음 (p = 0.23). 라벨 쌍 1개, 20 에피소드로 검정력 부족 — 결론 내리기 어려움.
- **측정 버그 수정:** K2 계산에서 probe가 두 타일에 정확히 같은 점수를 줄 때 둘 다 오답 처리하던 것을 동전 던지기(=chance)로 수정.
  사진 세트에서는 같은 사진 쌍의 두 질문이 neutral에서 완전히 같은 장면이라 동점이 생김. 위 표는 수정 후 값. 그림 파일럿 결론(GO)은 수정 후에도 동일.
- **분석 job 버그:** 자동 분석 job이 로그 폴더가 없어 실패 → 직접 실행 (스크립트 수정됨).

**종합 해석:** 그림에서 본 "K는 decode되지만 G가 손실" 패턴은 실사 State/Attribute에서 재현되지 않았다. 가장 가능성 높은 설명은
그림 파일럿의 K가 상당 부분 *질문 텍스트의 범주 정보*(예: insect → bee)였고, 실사 상태/재질 질문(예: "Which door keeps the room private?")에서는
그런 표면 단서가 없어서 K가 나타나지 않는다는 것. 행동 수준의 grounding 실패(정답 이름을 알려줘도 chance)는 모든 세트에서 일관됨.
다음 단계 제안: (1) 질문 텍스트만으로 K가 decode되는지 확인하는 text-only 대조군, (2) 질문 수·seed를 늘린 Emotion 재실행 (CEED 등 얼굴 데이터),
(3) 질문 초안 팀 검토 (`photo_output/*_questions.csv`).
