# Semantic KD와 Semantic Bridge (Ours) 학습 방법

두 방법 모두 OpenVLA SFT를 시작점으로 LoRA를 학습한다. 아래는 **현재 구현된 학습 방식**이다. `analysis/expected/`의 예상 수치는 실제 학습 결과가 아니다.

## 데이터 분할

두 VQA 데이터셋 모두 10개 카테고리 × 100개 pair이며, pair마다 좌·우 focus 2개 × 질문 5개가 있다. 분할은 **pair 단위 80/10/10**으로 하므로 같은 pair의 질문과 시뮬레이션 에피소드는 서로 다른 split에 섞이지 않는다.

| 데이터 단위 | Train | Validation | Test | 전체 |
| --- | ---: | ---: | ---: | ---: |
| 이미지 pair | 800 | 100 | 100 | 1,000 |
| VQA 질문 (pair당 10개) | 8,000 | 1,000 | 1,000 | 10,000 |
| 시뮬레이션 에피소드 (pair당 원본·좌우 교환 각 1개) | 1,600 | 200 | 200 | 2,000 |
| Robot grounding 질문 (에피소드 첫 프레임당 1개) | 1,600 | 200 | 200 | 2,000 |

Action 학습 샘플은 에피소드의 **행동 프레임마다 1개**여서 총수는 에피소드별 길이에 따라 달라진다. 학습 준비 코드는 성공한 에피소드만 사용하므로, 누락되거나 실패한 에피소드가 있으면 시뮬레이션·grounding 수는 표보다 적어진다.

## Semantic KD

**목표:** Prismatic VLM의 의미 판단을 OpenVLA에 전달하면서, 시뮬레이션 화면에서 정답 위치를 찾고 로봇 행동도 학습한다.

| 학습 항목 | 입력 | 기대 출력 |
| --- | --- | --- |
| VQA | 좌우 이미지를 나란히 붙인 pair 이미지 + 5종 VQA 중 한 질문 | 해당 질문의 정답 텍스트(`A`/`B` 또는 `Left`/`Right` 등) |
| Robot grounding | 시뮬레이션 첫 프레임 + 의미 추론 instruction + “Which side is the correct destination for the cube?” | `Left` 또는 `Right` |
| Action | 각 시뮬레이션 프레임 + “What action should the robot take to …?” | 정규화한 7차원 행동을 나타내는 OpenVLA action token 7개 |

한 업데이트에서 action 샘플 하나와 보조 샘플 하나를 사용한다. 보조 샘플은 VQA가 75%, robot grounding이 25% 확률로 선택된다.

$$
L_{\mathrm{KD}}=\frac{1}{|K|}\sum_{l\in K}\bigl(1-\cos(h_l^{S},h_l^{T})\bigr),
\qquad
L=L_{\mathrm{action}}+
\begin{cases}
0.5\,(L_{\mathrm{VQA}}+0.1\,gL_{\mathrm{KD}}), & \text{VQA 샘플}\;(75\%)\\
0.5\,(L_{\mathrm{ground}}+0.1\,gL_{\mathrm{KD}}), & \text{grounding 샘플}\;(25\%).
\end{cases}
$$

- $L_{\mathrm{action}}$: 시범 행동의 7개 action token에 대한 교차 엔트로피(CE). 행동을 재현하도록 학습한다.
- $L_{\mathrm{VQA}}$: 질문 정답 token에 대한 CE. 이미지의 대상과 의미를 답하도록 학습한다.
- $L_{\mathrm{ground}}$: 로봇 화면에서 정답 쪽(`Left`/`Right`)을 답하는 CE. 의미 판단을 화면의 위치에 연결한다.
- $L_{\mathrm{KD}}$: 같은 이미지·질문을 본 학생(OpenVLA)과 교사(Prismatic)의 **정답 생성 직전 hidden state** 사이 코사인 거리. 현재 $K=\{16\}$이다. $g=1$은 교사의 생성 답이 정답일 때, 아니면 $g=0$이다. 따라서 교사가 틀려도 정답 CE는 적용되고 KD 항만 빠진다. Action에는 KD를 적용하지 않는다.

## Semantic Bridge (Ours)

**목표:** 첫 화면에서 내린 `Left`/`Right` 판단을 이후 행동의 명시적 입력으로 고정해, 의미 판단이 action token으로 이어지게 한다. 별도 교사나 KD loss는 사용하지 않는다.

| 학습 항목 | 입력 | 기대 출력 |
| --- | --- | --- |
| Side decision | 시뮬레이션 첫 프레임 + instruction + “First give the target side, then the robot action.” | 정답 `Left` 또는 `Right` |
| Action | 각 시뮬레이션 프레임 + 같은 instruction + 정답 side 접두사와 `Action:` 표시 | OpenVLA action token 7개 |
| VQA | 시뮬레이션 화면 + 5종 VQA 중 한 질문 | 질문의 정답 텍스트 |

Policy 출력 형식은 모든 프레임에서 `Left/Right` → `Action:` → action token 7개다. 학습에서는 첫 프레임에만 side token CE를 적용하고, 후속 프레임에는 정답 side를 접두사로 제공하여 action CE만 적용한다. 추론 때는 첫 프레임에서 side를 **한 번** 고른 뒤 에피소드 내내 고정한다.

$$
L_{\mathrm{warmup}}=L_{\mathrm{side}}+0.5L_{\mathrm{VQA}},
\qquad
L_{\mathrm{joint}}=L_{\mathrm{action}}+
2\,\mathbf{1}_{t=0}L_{\mathrm{side}}+0.5L_{\mathrm{VQA}}.
$$

- $L_{\mathrm{side}}$: 첫 프레임의 정답 방향 token CE. Robot grounding을 policy 출력 안에서 직접 학습하는 항이다.
- $L_{\mathrm{action}}$: 정답 방향 뒤에 이어지는 7개 action token의 CE. 첫 프레임과 후속 프레임 모두에 적용된다.
- $L_{\mathrm{VQA}}$: 별도 VQA 질문의 정답 token CE. 이미지 의미를 읽는 능력을 보조한다.
- Warmup은 400업데이트이며 action loss를 제외한다. 이후 joint 학습은 3 epoch × 1,200업데이트다. Joint 단계에서 첫 프레임을 50% 확률로 뽑으므로, 그때의 side loss에 $1/0.5=2$를 곱한다.
