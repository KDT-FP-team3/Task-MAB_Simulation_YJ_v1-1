"""
Multi-Armed Bandit 실험
=======================
5개 슬롯머신에서 최적의 머신을 찾는 강화학습 실험
전략: Random, Greedy, ε-greedy (ε=0.1, ε=0.01)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# ── 한글 폰트 설정 ──────────────────────────────────────
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 시드 고정 (재현성) ─────────────────────────────────
np.random.seed(42)

# ── 저장 경로 ───────────────────────────────────────────
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
# 1. 환경 설계 — 슬롯머신(Bandit) 클래스
# ═══════════════════════════════════════════════════════
class SlotMachine:
    """하나의 슬롯머신을 나타냄. 정규분포 보상을 지급."""
    def __init__(self, mean_reward: float, std: float = 1.0):
        self.mean_reward = mean_reward   # 진짜 평균 (에이전트는 모름)
        self.std = std

    def pull(self) -> float:
        """레버를 당기면 정규분포 N(mean, std) 에서 보상 반환."""
        return np.random.normal(self.mean_reward, self.std)


class MultiArmedBandit:
    """여러 슬롯머신으로 구성된 카지노 환경."""
    def __init__(self, true_means: list[float]):
        self.machines = [SlotMachine(m) for m in true_means]
        self.k = len(true_means)           # 머신 수
        self.best_arm = int(np.argmax(true_means))  # 정답 머신

    def pull(self, arm: int) -> float:
        return self.machines[arm].pull()


# ═══════════════════════════════════════════════════════
# 2. 전략(Agent) 구현
# ═══════════════════════════════════════════════════════
class Agent:
    """에이전트 기본 클래스."""
    def __init__(self, k: int, name: str):
        self.k = k
        self.name = name
        self.counts = np.zeros(k)          # 머신별 선택 횟수
        self.values = np.zeros(k)          # 머신별 추정 평균 보상
        self.total_reward = 0.0
        self.reward_history = []           # 매 스텝 누적 보상

    def select_arm(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        # 점진적 평균 업데이트: Q ← Q + 1/n * (r - Q)
        self.values[arm] += (reward - self.values[arm]) / n
        self.total_reward += reward
        self.reward_history.append(self.total_reward)


class RandomAgent(Agent):
    """Random 전략: 매번 랜덤하게 머신을 선택."""
    def __init__(self, k):
        super().__init__(k, "Random")

    def select_arm(self):
        return np.random.randint(self.k)


class GreedyAgent(Agent):
    """Greedy 전략: 현재 추정값이 가장 높은 머신만 선택."""
    def __init__(self, k):
        super().__init__(k, "Greedy")

    def select_arm(self):
        return int(np.argmax(self.values))


class EpsilonGreedyAgent(Agent):
    """ε-greedy 전략: 확률 ε로 탐색, 1-ε로 활용."""
    def __init__(self, k, epsilon: float):
        super().__init__(k, f"ε-greedy (ε={epsilon})")
        self.epsilon = epsilon

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)     # Exploration
        else:
            return int(np.argmax(self.values))    # Exploitation


# ═══════════════════════════════════════════════════════
# 3. 실험 실행
# ═══════════════════════════════════════════════════════
TRUE_MEANS = [0.2, 0.5, 0.1, 0.8, 0.3]
N_STEPS = 1000
N_RUNS = 200  # 여러 번 반복해서 평균

def run_experiment(agent_class, bandit_means, n_steps, n_runs, **kwargs):
    """에이전트를 n_runs번 실험하여 평균 결과 반환."""
    k = len(bandit_means)
    all_rewards = np.zeros(n_steps)
    all_counts = np.zeros(k)
    all_optimal = np.zeros(n_steps)

    env_temp = MultiArmedBandit(bandit_means)
    best_arm = env_temp.best_arm

    for _ in range(n_runs):
        env = MultiArmedBandit(bandit_means)
        agent = agent_class(k, **kwargs)

        for t in range(n_steps):
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)
            all_rewards[t] += reward
            all_optimal[t] += (1 if arm == best_arm else 0)

        all_counts += agent.counts

    # 평균
    avg_rewards = np.cumsum(all_rewards / n_runs)
    avg_counts = all_counts / n_runs
    avg_optimal = np.cumsum(all_optimal / n_runs) / (np.arange(n_steps) + 1) * 100

    return avg_rewards, avg_counts, avg_optimal, agent.name


# ── 실험 ──
experiments = [
    (RandomAgent, {}),
    (GreedyAgent, {}),
    (EpsilonGreedyAgent, {"epsilon": 0.1}),
    (EpsilonGreedyAgent, {"epsilon": 0.01}),
]

results = []
for agent_cls, kw in experiments:
    cum_rewards, counts, optimal_pct, name = run_experiment(
        agent_cls, TRUE_MEANS, N_STEPS, N_RUNS, **kw
    )
    results.append((name, cum_rewards, counts, optimal_pct))
    print(f"[{name:25s}]  누적보상={cum_rewards[-1]:.1f}  머신별선택={counts}")


# ═══════════════════════════════════════════════════════
# 4. 시각화
# ═══════════════════════════════════════════════════════
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# ── (1) 슬롯머신 분포 시각화 ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(-3, 4, 300)
for i, mu in enumerate(TRUE_MEANS):
    y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - mu) ** 2)
    ax.fill_between(x, y, alpha=0.3, label=f'머신 {i+1} (μ={mu})')
    ax.axvline(mu, linestyle='--', alpha=0.5)
ax.set_title('각 슬롯머신의 보상 분포 (정규분포)', fontsize=16, fontweight='bold')
ax.set_xlabel('보상', fontsize=13)
ax.set_ylabel('확률밀도', fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig1_reward_distribution.png'), dpi=150)
plt.close()

# ── (2) 누적 보상 곡선 ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for i, (name, cum_rew, _, _) in enumerate(results):
    ax.plot(cum_rew, label=name, linewidth=2, color=colors[i])
ax.set_title('전략별 누적 보상 비교', fontsize=16, fontweight='bold')
ax.set_xlabel('선택 횟수 (스텝)', fontsize=13)
ax.set_ylabel('누적 보상', fontsize=13)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig2_cumulative_reward.png'), dpi=150)
plt.close()

# ── (3) 머신별 선택 비율 (전략별) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
machine_labels = [f'머신{i+1}\n(μ={m})' for i, m in enumerate(TRUE_MEANS)]

for idx, (name, _, counts, _) in enumerate(results):
    ax = axes[idx]
    total = counts.sum()
    pcts = counts / total * 100
    bars = ax.bar(machine_labels, pcts, color=colors[idx], edgecolor='white', linewidth=1.5)
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.set_ylabel('선택 비율 (%)', fontsize=11)
    ax.set_ylim(0, 105)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')
    # 최적 머신 강조
    best_idx = np.argmax(TRUE_MEANS)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('전략별 머신 선택 비율', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig3_selection_ratio.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── (4) 최적 머신 선택률 (시간에 따른 변화) ─────────
fig, ax = plt.subplots(figsize=(10, 6))
for i, (name, _, _, opt_pct) in enumerate(results):
    ax.plot(opt_pct, label=name, linewidth=2, color=colors[i])
ax.set_title('최적 머신(머신4) 선택률 변화', fontsize=16, fontweight='bold')
ax.set_xlabel('선택 횟수 (스텝)', fontsize=13)
ax.set_ylabel('최적 선택률 (%)', fontsize=13)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig4_optimal_selection.png'), dpi=150)
plt.close()

# ── (5) ε 비교 실험 (0.01 vs 0.1 vs 0.3 vs 0.5) ────
epsilons = [0.01, 0.1, 0.3, 0.5]
eps_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for eps, col in zip(epsilons, eps_colors):
    cum_r, _, opt_p, nm = run_experiment(
        EpsilonGreedyAgent, TRUE_MEANS, N_STEPS, N_RUNS, epsilon=eps
    )
    ax1.plot(cum_r, label=f'ε={eps}', linewidth=2, color=col)
    ax2.plot(opt_p, label=f'ε={eps}', linewidth=2, color=col)

ax1.set_title('ε 값에 따른 누적 보상', fontsize=14, fontweight='bold')
ax1.set_xlabel('스텝'); ax1.set_ylabel('누적 보상')
ax1.legend(fontsize=11); ax1.grid(alpha=0.3)

ax2.set_title('ε 값에 따른 최적 선택률', fontsize=14, fontweight='bold')
ax2.set_xlabel('스텝'); ax2.set_ylabel('최적 선택률 (%)')
ax2.legend(fontsize=11); ax2.grid(alpha=0.3)

plt.suptitle('ε(엡실론) 값 변화 실험', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig5_epsilon_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── (6) 개념 요약 다이어그램 ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis('off')

# 제목
ax.text(5, 9.5, 'Multi-Armed Bandit 핵심 개념', fontsize=22, fontweight='bold',
        ha='center', va='center', color='#2c3e50')

# Exploration vs Exploitation 박스
# 탐색
from matplotlib.patches import FancyBboxPatch
box_exp = FancyBboxPatch((0.5, 5.5), 3.5, 3, boxstyle="round,pad=0.3",
                          facecolor='#3498db', alpha=0.15, edgecolor='#3498db', linewidth=2)
ax.add_patch(box_exp)
ax.text(2.25, 8.0, '[탐색] Exploration', fontsize=15, fontweight='bold',
        ha='center', color='#2980b9')
ax.text(2.25, 7.2, '"새로운 머신도 시도해보자"', fontsize=11, ha='center', style='italic')
ax.text(2.25, 6.5, '• 정보 수집이 목적\n• 단기적으로 손해 가능\n• 장기적으로 더 좋은 선택 발견',
        fontsize=10, ha='center', va='top')

# 활용
box_ext = FancyBboxPatch((5.5, 5.5), 3.5, 3, boxstyle="round,pad=0.3",
                          facecolor='#e74c3c', alpha=0.15, edgecolor='#e74c3c', linewidth=2)
ax.add_patch(box_ext)
ax.text(7.25, 8.0, '[활용] Exploitation', fontsize=15, fontweight='bold',
        ha='center', color='#c0392b')
ax.text(7.25, 7.2, '"지금까지 가장 좋았던 걸 선택"', fontsize=11, ha='center', style='italic')
ax.text(7.25, 6.5, '• 즉각적 보상이 목적\n• 안전한 선택\n• 더 좋은 선택 놓칠 수 있음',
        fontsize=10, ha='center', va='top')

# 화살표와 균형
ax.annotate('', xy=(5.3, 7.0), xytext=(4.2, 7.0),
            arrowprops=dict(arrowstyle='<->', color='#f39c12', lw=3))
ax.text(4.75, 7.5, '균형!', fontsize=14, fontweight='bold', ha='center', color='#f39c12')

# 전략 비교 테이블
strategies = [
    ('Random',         '항상 랜덤', '탐색 100%', '#95a5a6'),
    ('Greedy',         '항상 최고만', '활용 100%', '#e74c3c'),
    ('ε-greedy',       'ε% 탐색 + (1-ε)% 활용', '균형 조절', '#2ecc71'),
]
y_start = 4.5
ax.text(5, y_start + 0.3, '▼ 전략 비교', fontsize=14, fontweight='bold', ha='center', color='#2c3e50')
for i, (name, desc, ratio, col) in enumerate(strategies):
    y = y_start - 1.0 - i * 1.0
    box = FancyBboxPatch((1, y-0.3), 8, 0.8, boxstyle="round,pad=0.15",
                          facecolor=col, alpha=0.12, edgecolor=col, linewidth=1.5)
    ax.add_patch(box)
    ax.text(2.2, y+0.1, name, fontsize=12, fontweight='bold', color=col, ha='center')
    ax.text(5.2, y+0.1, desc, fontsize=11, ha='center')
    ax.text(8.2, y+0.1, ratio, fontsize=10, ha='center', color='#7f8c8d')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'fig6_concept_summary.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\n[OK] All charts saved!")
print(f"   Location: {SAVE_DIR}")
