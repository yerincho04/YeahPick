"""Generate a factual Korean experiment report from persisted training records."""
import argparse
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
c = json.loads((a.run/'run_config.json').read_text())
d = json.loads(a.audit.read_text())
done = json.loads((a.run/'completion.json').read_text())
assert done['status'] == 'completed' and done['updates'] == c['total_updates']
metrics = [json.loads(x) for x in (a.run/'metrics.jsonl').read_text().splitlines()]
assert len(metrics) == done['updates']
assert [x['update'] for x in metrics] == list(range(1, done['updates']+1))
mean = lambda xs: statistics.mean(x['loss'] for x in xs)
rows = '\n'.join(f'| {cat} | {n} |' for cat,n in d['by_category'].items())
epochs = '\n'.join(f'| {e} | {mean([x for x in metrics if e-1 < x["epoch"] <= e]):.6f} |' for e in range(1,c['epochs']+1))
report = f'''# OpenVLA VQA 파인튜닝 실험 보고서

## 목적과 범위

공식 OpenVLA의 semantic reasoning 관련 표현을 VQA 텍스트 지도학습으로
적응시키는 첫 실험이다. 로봇 행동 개선은 후속 평가 가설이며, 이번 학습으로
입증한 결과가 아니다. 사용자 요청에 따라 전체 데이터를 학습에 사용했고
검증·테스트 분할, VQA 정답률 측정, 로봇 행동 평가는 수행하지 않았다.

## 모델과 환경

- Base: `{c['model_id']}`
- Base revision: `{c['revision']}`
- VESSL: CS-LEC-05 / hyerin-vulkan-50g, yonsei-ai-gpu, ai-gpu-07
- GPU: RTX 3090 24GiB × 2, DDP
- 정밀도: {c['precision']}
- 정확한 패키지 버전: `environment.txt`

## 데이터

{d['groups']:,}개 VQA 묶음에서 질문 다섯 개씩을 펼쳐 {d['examples']:,}개 QA를
사용했다. 각 epoch에 모든 QA를 정확히 한 번씩 사용했다. 총 학습 노출 횟수는
{d['examples']*c['epochs']:,}회다. 합계 {d['pairs']}개 좌우 이미지 쌍을 사용했다.

| 카테고리 | QA 수 |
|---|---:|
{rows}

Object recognition, Attribute recognition, Semantic reasoning, Visual grounding,
Counterfactual grounding은 각각 1,600개다. 카테고리·유형에 추가 가중치를 주지 않았다.
원본 branch/commit/file/해시는 `data_audit.json`에 기록했다. jaeho의 통합 JSONL과
카테고리 JSONL은 같은 QA이므로 카테고리 파일만 읽었다. 원본 질문 풀과 pair
metadata는 추가 학습 레코드로 세지 않았다. 이미지가 없는 KDEF는 사용자 승인으로
제외했다. 나머지 8개 카테고리에서는 샘플을 제외하거나 새 정답을 만들지 않았다.

입력은 좌우 두 타일을 고정 순서로 배치한 224×224 RGB 이미지다. 각 타일은
112×224 영역 안에 종횡비를 유지해 배치하고 빈 공간을 흰색으로 채웠다.
정답·labels·knowledge·카테고리 정보는 입력에 포함하지 않았다. 이미지 증강 및
텍스트 잘림은 적용하지 않았다. 이 작은 해상도는 세부 시각 정보의 제약이 될 수 있다.

Train manifest SHA-256: `{d['train_sha256']}`

## 학습 방법

- LoRA: LLM attention q_proj/v_proj, rank 16, alpha 32, dropout 0.05
- 학습 파라미터: {c['trainable_parameters']:,}개
- Vision encoder, projector, embedding, LM head 동결
- 정답 텍스트와 EOS에만 cross-entropy, 질문·시각 토큰·padding은 -100 마스킹
- AdamW, LR {c['lr']}, weight decay 0.01, gradient clipping 1.0
- Cosine LR schedule, 전체 update의 5% warmup
- Per-GPU batch {c['batch_size']}, accumulation {c['accumulation']}, effective batch {c['effective_batch']}
- {c['epochs']} epoch, {done['updates']:,} optimizer updates, seed {c['seed']}
- LLM gradient checkpointing, SDPA attention, BF16; 4bit 양자화 사용 안 함

짧은 별도 smoke 실행으로 loss/gradient의 유한성과 학습 가능 여부를 확인한 뒤,
본 학습은 원본 모델에서 새로 시작했다. Smoke 실행 결과는 본 학습 지표에 섞지 않았다.

## 실제 학습 결과

- 완료 상태: `{done['status']}`
- 첫 10 update 평균 train loss: {mean(metrics[:10]):.6f}
- 마지막 10 update 평균 train loss: {mean(metrics[-10:]):.6f}
- 전체 평균 train loss: {mean(metrics):.6f}
- 기록된 rank-0 최대 GPU allocated memory: {max(x['peak_gpu_gib'] for x in metrics):.3f} GiB
- 학습 루프·epoch 저장 소요 시간: {done['elapsed_seconds']/60:.2f}분

| Epoch | 평균 train loss |
|---|---:|
{epochs}

지표는 batch 내 정답 토큰 평균 loss를 gradient accumulation과 GPU에 대해 평균낸
학습 지표다. 검증 점수나 일반화 성능이 아니다. 검증 데이터를 사용하지 않았으므로
최종 3 epoch 모델을 저장했으며, best checkpoint라고 부르지 않는다.

## 산출물과 후속 사용

GitHub의 `hyerin` 브랜치 `vqa_finetuning/`에는 코드, 원본 모델 revision,
데이터 audit와 이미지 hash manifest, 설정, 패키지 목록, 학습 지표 및 본 보고서를
저장한다. Weight와 optimizer 상태는 GitHub에 올리지 않는다.

로컬 `outputs/vqa_finetuning/weights/`에 최종 LoRA adapter와 원본에 merge한
BF16 전체 모델을 보관한다. `export_merged.py`로 merge를 재현할 수 있다.
전체 모델은 원본 action normalization 설정을 유지한다. 학습에서 action loss를
사용하지 않았으므로 이후 행동 평가에서 기존 제어 능력 유지 여부와 의미 지시
성공률을 모두 확인해야 한다. 학습에 사용한 이미지/질문과의 중복 여부도 후속
평가 설계에서 구분해야 한다.
'''
a.output.write_text(report)
print(a.output)
