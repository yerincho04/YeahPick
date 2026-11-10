"""Answer-only LoRA SFT for official OpenVLA; all input QA used for training.

Launch with torchrun --standalone --nproc_per_node=2 train.py --data ... --output ...
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForVision2Seq, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

PREFIX = 'The image contains two tiles: the left half is the left tile and the right half is the right tile. Answer the question concisely. '


def encode(tokenizer, question, answer):
    prompt = 'In: ' + PREFIX + question.strip() + '\nOut:'
    prefix = tokenizer(prompt, add_special_tokens=True).input_ids
    full = tokenizer(prompt + ' ' + answer.strip(), add_special_tokens=True).input_ids
    # Fail rather than silently supervising prompt tokens if tokenizer behavior changes.
    assert full[:len(prefix)] == prefix, 'Prompt/answer tokenization boundary mismatch'
    full = full + [tokenizer.eos_token_id]
    labels = [-100]*len(prefix) + full[len(prefix):]
    assert len(full) == len(labels) and any(x != -100 for x in labels)
    assert len(full) + 256 <= 2048, 'Example exceeds model context; no truncation allowed'
    return full, labels


class VQA(Dataset):
    def __init__(self, root, processor):
        self.root, self.processor = root, processor
        self.rows = [json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()]
        self.tokens = [encode(processor.tokenizer, r['question'], r['answer']) for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.root/self.rows[i]['image']) as im:
            pixels = self.processor.image_processor(im.convert('RGB'), return_tensors='pt')['pixel_values'][0]
        ids, labels = self.tokens[i]
        return torch.tensor(ids), torch.tensor(labels), pixels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-updates', type=int, default=0, help='Smoke test only; 0 uses all epochs')
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    rank, world, local = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        dist.init_process_group('nccl')
    torch.cuda.set_device(local)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    a.output.mkdir(parents=True, exist_ok=True)
    base = json.loads((Path(__file__).parent/'base_model.json').read_text())
    processor = AutoProcessor.from_pretrained(base['model_id'], revision=base['revision'], trust_remote_code=True)
    processor.tokenizer.padding_side = 'right'
    dataset = VQA(a.data, processor)
    audit = json.loads((a.data/'data_audit.json').read_text())
    assert len(dataset) == audit['examples'] == 8000
    assert hashlib.sha256((a.data/'train.jsonl').read_bytes()).hexdigest() == audit['train_sha256']
    model = AutoModelForVision2Seq.from_pretrained(base['model_id'], revision=base['revision'],
        trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa').to(local)
    model.config.use_cache = False
    model.language_model.config.use_cache = False
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, revision=base['revision'],
        target_modules=r'language_model\.model\.layers\.\d+\.self_attn\.(q_proj|v_proj)', bias='none'))
    trainable = [v for v in model.parameters() if v.requires_grad]
    # Keep adapter/optimizer updates in FP32; the frozen base and compute use BF16.
    for parameter in trainable:
        parameter.data = parameter.data.float()
    assert trainable, 'No LoRA parameters selected'
    names = [k for k,v in model.named_parameters() if v.requires_grad]
    assert all('language_model' in k and 'lora_' in k for k in names)
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=a.seed, drop_last=False)

    def collate(batch):
        ids, labels, pixels = zip(*batch)
        ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(input_ids=ids, labels=labels, attention_mask=ids.ne(processor.tokenizer.pad_token_id), pixel_values=torch.stack(pixels))

    loader = DataLoader(dataset, batch_size=a.batch_size, sampler=sampler, collate_fn=collate, num_workers=0)
    # Exact batches: no repeated distributed padding or dropped samples in this experiment.
    assert len(dataset) % (world*a.batch_size*a.accumulation) == 0
    per_epoch = len(loader)//a.accumulation
    total = per_epoch*a.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(total*0.05)), total)
    config = dict(vars(a), **base, world_size=world, examples=len(dataset),
        effective_batch=world*a.batch_size*a.accumulation, total_updates=total,
        trainable_parameters=sum(v.numel() for v in trainable), trainable_names=names,
        precision='BF16 frozen base/compute; FP32 adapters/optimizer; not quantized', target='LLM q_proj/v_proj only; vision/projector frozen',
        supervision='Answer and EOS only; prompt, visual tokens, padding masked',
        data_sha256=hashlib.sha256((a.data/'train.jsonl').read_bytes()).hexdigest())
    start_epoch, updates = 0, 0
    if a.resume:
        assert a.resume.resolve().parent == a.output.resolve(), 'Resume into the original output directory'
        previous = json.loads((a.output/'run_config.json').read_text())
        for key in ['model_id', 'revision', 'world_size', 'data_sha256', 'epochs', 'batch_size', 'accumulation', 'lr', 'seed']:
            assert previous[key] == config[key], 'Resume configuration changed: '+key
        state = torch.load(a.resume/'training_state.pt', map_location='cpu')
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(a.resume/'adapter_model.safetensors'))
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_epoch, updates = state['next_epoch'], state['updates']
        rng = state['rng'][rank]
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'], local)
        if rank == 0:
            # Discard log records from an interrupted, unsaved partial epoch.
            old_metrics = [json.loads(x) for x in (a.output/'metrics.jsonl').read_text().splitlines()]
            old_metrics = [x for x in old_metrics if x['update'] <= updates]
            assert [x['update'] for x in old_metrics] == list(range(1, updates+1))
            (a.output/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in old_metrics))
    if rank == 0:
        (a.output/'run_config.json').write_text(json.dumps(config, indent=2, default=str)+'\n')
        print(json.dumps(config, default=str), flush=True)
    ddp = DDP(model, device_ids=[local], find_unused_parameters=False) if world > 1 else model
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    finished = False
    for epoch in range(start_epoch, a.epochs):
        sampler.set_epoch(epoch)
        ddp.train(); model.base_model.model.vision_backbone.eval(); model.base_model.model.projector.eval()
        running = torch.zeros((), device=local)
        for i, batch in enumerate(loader):
            boundary = (i+1) % a.accumulation == 0
            sync = nullcontext() if boundary or world == 1 else ddp.no_sync()
            batch = {k: v.to(local, dtype=torch.bfloat16 if k == 'pixel_values' else v.dtype) for k,v in batch.items()}
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                loss = ddp(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch={epoch} batch={i}')
                scaled = loss/a.accumulation
                running += scaled.detach()
                scaled.backward()
            if boundary:
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError('Non-finite gradient')
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                updates += 1
                if world > 1:
                    dist.all_reduce(running); running /= world
                if rank == 0:
                    entry = dict(update=updates, epoch=epoch+(i+1)/len(loader), loss=running.item(),
                        learning_rate=scheduler.get_last_lr()[0], grad_norm=norm.item(), elapsed_seconds=time.time()-start,
                        peak_gpu_gib=torch.cuda.max_memory_allocated(local)/2**30)
                    with (a.output/'metrics.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
                    if updates % 10 == 0 or updates == 1: print(json.dumps(entry), flush=True)
                running.zero_()
                if a.max_updates and updates >= a.max_updates:
                    finished = True; break
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(local))
        all_rng = [None]*world
        if world > 1: dist.all_gather_object(all_rng, rng)
        else: all_rng = [rng]
        if rank == 0:
            target = a.output/('smoke' if a.max_updates else f'epoch-{epoch+1}')
            target.mkdir(exist_ok=True)
            model.save_pretrained(target, safe_serialization=True)
            processor.save_pretrained(target)
            # Resume supports completed epochs only. Smoke checkpoints are NOT resumable.
            if not a.max_updates:
                torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    next_epoch=epoch+1, updates=updates, rng=all_rng), target/'training_state.pt')
            if epoch == a.epochs-1 or finished:
                model.save_pretrained(a.output/'final_adapter', safe_serialization=True)
                processor.save_pretrained(a.output/'final_adapter')
                (a.output/'completion.json').write_text(json.dumps(dict(status='smoke_completed' if a.max_updates else 'completed',
                    updates=updates, epochs_completed=(epoch+1 if not a.max_updates else None),
                    elapsed_seconds=time.time()-start, evaluated=False), indent=2)+'\n')
        if world > 1: dist.barrier()
        if finished: break
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()
