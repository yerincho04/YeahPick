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
