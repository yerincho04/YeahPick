# Semantic KD 테스트 결과

테스트는 학습에 쓰지 않은 pair 100개(10개 카테고리)를 사용했다. 각 pair마다 VQA 10문항, 로봇 rollout 2회, 로봇 화면 grounding 2문항을 평가했다. 비교 대상은 KD를 적용하지 않은 **OpenVLA RLVLA SFT 체크포인트**다.

## 1. VQA와 로봇 rollout

**핵심 결론: VQA는 70.6% 맞혔지만 로봇은 정답 타일을 51% 골랐다. 질문에 답하는 능력이 곧 행동 성공을 뜻하지는 않는다.**

각 칸은 해당 카테고리·질문 유형의 정답률이다. 질문 유형별로 카테고리당 20문항, 로봇 rollout은 카테고리당 20회다.

| Category | Object recognition | Attribute recognition | Semantic reasoning | Visual grounding | Counterfactual grounding | VQA 평균 | 로봇 성공 |
|---|---:|---:|---:|---:|---:|---:|---:|
| attribute | 95% | 65% | 60% | 65% | 65% | 70% | 60% (12/20) |
| celebrity | 95% | 85% | 80% | 85% | 70% | 83% | 55% (11/20) |
| counting | 80% | 55% | 45% | 60% | 65% | 61% | 40% (8/20) |
| emotion | 100% | 80% | 40% | 45% | 45% | 62% | 50% (10/20) |
| living_world | 80% | 85% | 85% | 85% | 90% | 85% | 75% (15/20) |
| public_info | 65% | 55% | 25% | 55% | 55% | 51% | 45% (9/20) |
| state | 95% | 55% | 50% | 70% | 70% | 68% | 25% (5/20) |
| symmetry | 70% | 80% | 50% | 35% | 40% | 55% | 60% (12/20) |
| time_clock | 100% | 100% | 100% | 100% | 100% | 100% | 50% (10/20) |
| traffic | 80% | 70% | 85% | 70% | 50% | 71% | 50% (10/20) |
| **전체 평균** | **86% (172/200)** | **73% (146/200)** | **62% (124/200)** | **67% (134/200)** | **65% (130/200)** | **70.6% (706/1000)** | **51% (102/200)** |

`Object recognition`(86%)이 가장 높고 `Semantic reasoning`(62%)이 가장 낮다. 특히 `time_clock`은 VQA를 모두 맞혔지만 로봇 성공은 절반이다.

## 2. OpenVLA SFT와 비교

**핵심 결론: KD는 OpenVLA SFT보다 로봇 성공률이 8.5%p 높다. 하지만 `state`처럼 오히려 크게 나빠진 카테고리도 있어 개선이 고르지는 않다.**

성공 여부는 마지막 프레임에서 큐브가 정답 타일 영역에 있는지로 판정했다. 큐브를 완벽하게 내려놓았는지는 별도로 요구하지 않는다.

| 카테고리 | KD 성공 | OpenVLA SFT 성공 | KD − SFT |
|---|---:|---:|---:|
| attribute | 12/20 (60%) | 7/20 (35%) | +25%p |
| celebrity | 11/20 (55%) | 10/20 (50%) | +5%p |
| counting | 8/20 (40%) | 9/20 (45%) | −5%p |
| emotion | 10/20 (50%) | 8/20 (40%) | +10%p |
| living_world | 15/20 (75%) | 9/20 (45%) | +30%p |
| public_info | 9/20 (45%) | 9/20 (45%) | 0%p |
| state | 5/20 (25%) | 12/20 (60%) | −35%p |
| symmetry | 12/20 (60%) | 7/20 (35%) | +25%p |
| time_clock | 10/20 (50%) | 7/20 (35%) | +15%p |
| traffic | 10/20 (50%) | 7/20 (35%) | +15%p |
| **전체** | **102/200 (51%)** | **85/200 (42.5%)** | **+8.5%p** |

| 추가 지표 | KD | OpenVLA SFT |
|---|---:|---:|
| 어느 타일이든 답한 비율 (`answered`) | 200/200 (100%) | 182/200 (91%) |
| 원본 배치 성공 | 50/100 (50%) | 41/100 (41%) |
| 좌우 교환 성공 | 52/100 (52%) | 44/100 (44%) |
| 자유 생성 VQA 채점 | 706/1000 (70.6%) | 0/1000 (0%) |

`answered`는 큐브가 정답 또는 오답 타일 중 하나에 있는 비율이다. SFT의 VQA 0%는 유효한 선택지 답을 생성하지 못한 결과이므로, 의미를 전혀 모른다는 뜻은 아니다.

## 3. VQA 답이 로봇 선택으로 이어졌는가?

**핵심 결론: `Semantic reasoning`을 맞혀도 로봇 정답률은 거의 그대로다. 좌우를 바꾸면 KD도 과반수에서 물체보다 화면 위치를 따라간다.**

각 pair에서 Knowledge 지시의 정답 물체에 관한 `Semantic reasoning` VQA 1문항을 골라, **원래 배치**의 로봇 결과와 비교했다. 총 100개 pair다.

| Semantic reasoning VQA | 로봇 정답 타일 | 로봇 오답 타일 | 로봇 정답률 |
|---|---:|---:|---:|
| 정답 | 30 | 29 | 30/59 (50.8%) |
| 오답 | 20 | 21 | 20/41 (48.8%) |

VQA를 맞힌 경우와 틀린 경우의 로봇 정답률 차이는 약 **2%p**다.

| 좌우 교환 시 선택의 일관성 | KD | OpenVLA SFT |
|---|---:|---:|
| 두 배치에서 같은 의미의 물체를 선택 | 38/100 (38%) | 15/85 (17.6%) |
| 두 배치에서 같은 화면상 위치를 선택 | 62/100 (62%) | 70/85 (82.4%) |
| 두 배치 중 하나라도 타일에 답하지 않음 | 0/100 | 15/100 |

SFT의 앞 두 비율은 두 배치 모두 타일에 답한 85개 pair에서만 계산했다. `time_clock`은 정답 물체에 관한 `Semantic reasoning` VQA가 10/10이었지만, 좌우 교환 후 같은 물체를 고른 pair는 0/10이었다.

## 4. Robot grounding

**핵심 결론: 로봇 화면에서 정답 타일을 직접 묻자 54%만 맞혔고, 답의 75%가 `Left`였다. 행동 이전의 화면 판단부터 좌우 편향이 보인다.**

성공한 scripted 시연의 첫 로봇 카메라 프레임을 보여주고, Knowledge 지시의 정답 위치를 `Left`/`Right`로 직접 답하게 했다.

| 카테고리 | 정답 수 / 20 | 정답률 |
|---|---:|---:|
| attribute | 12/20 | 60% |
| celebrity | 8/20 | 40% |
| counting | 10/20 | 50% |
| emotion | 14/20 | 70% |
| living_world | 17/20 | 85% |
| public_info | 9/20 | 45% |
| state | 7/20 | 35% |
| symmetry | 11/20 | 55% |
| time_clock | 10/20 | 50% |
| traffic | 10/20 | 50% |
| **전체** | **108/200** | **54%** |

| 배치 | 정답 수 / 100 | 정답률 |
|---|---:|---:|
| 원본 | 52/100 | 52% |
| 좌우 교환 | 56/100 | 56% |

| 로봇 화면 grounding | 로봇 정답 타일 | 로봇 오답 타일 | 로봇 정답률 |
|---|---:|---:|---:|
| 정답 | 66 | 42 | 66/108 (61.1%) |
| 오답 | 36 | 56 | 36/92 (39.1%) |

Grounding을 맞혔을 때 로봇도 맞힌 비율은 61.1%, 틀렸을 때 로봇이 맞힌 비율은 39.1%다. 정답 위치는 Left·Right 각각 100회인데, 모델 답은 **Left 150회·Right 50회**다. 정답이 Left일 때 79/100, Right일 때 29/100을 맞혔다. `time_clock`에서는 grounding 답이 모두 Left였지만 로봇의 마지막 선택은 모두 오른쪽이었다.

Grounding은 시연의 첫 프레임에서, 로봇 선택은 별도 rollout의 마지막 프레임에서 측정했다. 따라서 어느 내부 단계에서 판단이 달라졌는지는 알 수 없다. `grounding_teacher_forced_exact`는 정답을 입력에 넣고 계산한 값이라 여기의 직접 답변 정답률과 다르다. SFT grounding은 유효한 좌우 답을 내지 않아 0%였다고 보고되었지만, 예측 파일이 없어 세부 분석에는 넣지 않았다.

## 사용한 결과 파일

- KD: [`output/test_metrics.json`](output/test_metrics.json), [`output/test_vqa_predictions.jsonl`](output/test_vqa_predictions.jsonl), [`output/test_rollout_episodes.jsonl`](output/test_rollout_episodes.jsonl)
- OpenVLA SFT: [`output/test_metrics (1).json`](output/test_metrics%20(1).json), [`output/test_rollout_episodes (1).jsonl`](output/test_rollout_episodes%20(1).jsonl)
- 로봇 화면 grounding: [`output/test_grounding_metrics.json`](output/test_grounding_metrics.json), [`output/test_grounding_predictions.jsonl`](output/test_grounding_predictions.jsonl)
- Pair별 연결: [`output/pairwise_vqa_rollout.csv`](output/pairwise_vqa_rollout.csv)
