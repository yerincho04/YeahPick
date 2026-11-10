"""Render persisted train loss / learning-rate curves; no evaluation metrics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--metrics', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rows = [json.loads(x) for x in a.metrics.read_text().splitlines()]
x = [r['update'] for r in rows]
loss = [r['loss'] for r in rows]
smooth = [sum(loss[max(0,i-19):i+1])/len(loss[max(0,i-19):i+1]) for i in range(len(loss))]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, (top, bottom) = plt.subplots(2, 1, figsize=(10,6), sharex=True, gridspec_kw={'height_ratios':[3,1]}, constrained_layout=True)
top.plot(x, loss, color='#b2c2d2', linewidth=0.8, alpha=0.65, label='Per-update train loss')
top.plot(x, smooth, color='#185fba', linewidth=1.8, label='Trailing mean (20 updates)')
top.set_yscale('log'); top.set_ylabel('Answer-token cross-entropy (log scale)')
top.set_title('OpenVLA VQA LoRA training', loc='left', fontweight='bold')
top.legend(frameon=False, loc='upper right')
bottom.plot(x, [r['learning_rate'] for r in rows], color='#188570', linewidth=1.7)
bottom.set_ylabel('Learning rate'); bottom.set_xlabel('Optimizer update')
bottom.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
for ax in (top, bottom):
    ax.grid(alpha=0.15)
    for boundary in (500,1000): ax.axvline(boundary, color='#617080', alpha=0.4, linestyle='--', linewidth=0.8)
fig.suptitle('8,000 QA · 3 epochs · training metrics only; no validation/test evaluation', fontsize=10, color='#546271')
fig.savefig(a.output, dpi=180, facecolor='white')
plt.close(fig)
print(a.output)
