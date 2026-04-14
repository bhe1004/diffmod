# DiffMod: Hierarchical Modular Residual RL on Diffusion Policy

## 이 파일의 목적
Claude Code가 코드 결정을 내릴 때 참조하는 기준 문서다.
구현 전 반드시 읽고, 설계 의도에 맞게 코드를 작성할 것.

---

## 1. 핵심 주장 (코드 결정의 기준)

> "Frozen diffusion policy 위에 계층적 modular residual RL을 올리면,
>  (1) diffusion의 multi-modal 경로(CW/CCW)와
>  (2) task 단계(approach/reorient/translate/align)를
>  supervision 없이 자동으로 분리해서 포착한다."

이 주장을 검증하는 것이 모든 코드의 목적이다.
불필요한 추상화, 과도한 일반화는 하지 않는다.
분석 코드(UMAP, 클러스터링)는 구현 코드만큼 중요하다. 빠뜨리지 말 것.

### 선행 연구와의 차이 (포지셔닝)
- **ResiP** (Ankile et al., 2024): frozen diffusion + residual RL. 단, monolithic residual이라 multi-modal 경로를 구분하지 못하고 high randomization에서 성능 포화.
- **USR** (Zhu et al., ICLR 2026): latent steering으로 mode 선택 + residual refinement. 단, 여전히 단일 actor이고 어떤 module이 어떤 mode를 담당하는지 해석 불가.
- **DiffMod (우리)**: residual 자체를 계층적 modular 구조로 설계 → mode와 task 단계를 동시에 분리 + z_task/z_mode로 interpretable한 분석 가능.

---

## 2. 환경 및 기반 코드

### 사용 환경
- **Task**: Push-T (diffusion_policy 레포의 기존 환경 그대로 사용)
- **Base policy**: pretrained diffusion policy (low-dim, state-based)
- **Checkpoint 다운로드**:
```bash
mkdir -p data/
wget -P data/ https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt
```

### Push-T 스펙
```
state  (12d): pusher_xy (2) + block_pose (5: xy+sin+cos) + goal_pose (5: xy+sin+cos)
action  (2d): pusher xy velocity
chunk size Ta: 16 steps (diffusion이 한 번에 16스텝 예측)
episode_len: 300 steps
success criterion: block-goal coverage > 0.95
base policy success rate: ~0.969
```

### 기존 레포 파일 (절대 수정 금지)
```
diffusion_policy/env/pusht/pusht_env.py
diffusion_policy/policy/diffusion_unet_lowdim_policy.py
diffusion_policy/dataset/pusht_lowdim_dataset.py
diffusion_policy/common/pytorch_util.py
```

### 새로 만들 디렉토리 (모든 새 코드는 여기에)
```
diffusion_policy/diffmod/
```

---

## 3. 전체 아키텍처

```
state (12d)
    │
    ├─────────────────────────────┐
    ▼                             ▼
[High-level Modulation]     [Low-level Modulation]
입력: block_pose(5),         입력: pusher_vel(2),
      goal_pose(5),                pusher_to_block(2),
      block_to_goal(2)             traj_history(5*2=10)
→ MLP: 12→64→32→16          → MLP: 14→64→32→16
    │                             │
    ▼                             ▼
 z_task (16d)                 z_mode (16d)
 w_task = softmax(·) (4)      w_mode = softmax(·) (2)
    │                             │
    └──────────┬──────────────────┘
               ▼
   w_combined = outer_product(w_task, w_mode).flatten()
   shape: (B, 4*2=8)
               │
               ▼
    [Modular Base Network]
    구조: 12→64→64→2
    각 layer l: W_l_eff = Σ_ij (w_task_i · w_mode_j) · W_l^(i,j)
    총 파라미터 세트: M_task * M_mode = 4 * 2 = 8개
               │
               ▼
    raw_delta = base_net(state, w_combined)
    Δa = tanh(raw_delta) * 0.1        ← residual_scale=0.1 고정
               │
               ▼
    a_final = a_diffusion_chunk[t] + Δa   ← 매 스텝
```

### 핵심 설계 원칙
- Diffusion은 **완전히 frozen** (requires_grad=False)
- High-level modulation: lr=1e-4 (느리게, task 단계는 천천히 바뀜)
- Low-level modulation: lr=3e-4 (빠르게, mode는 초반에 빠르게 결정)
- Base network: lr=1e-4

---

## 4. 구현할 파일 목록

```
diffusion_policy/diffmod/
├── __init__.py
├── modular_network.py           # ModularLayer, ModularBaseNetwork
├── hierarchical_modulation.py   # HighLevelModulation, LowLevelModulation,
│                                #   HierarchicalModulation
├── modular_residual_policy.py   # ModularResidualPolicy (전체 wrapper)
├── ppo_trainer.py               # RolloutBuffer, PPOTrainer
├── pusht_diffmod_runner.py      # 환경 실행, rollout 수집
├── train_diffmod.py             # 학습 엔트리포인트 (argparse)
├── eval_diffmod.py              # 평가 엔트리포인트
└── analysis/
    ├── __init__.py
    ├── collect_embeddings.py    # rollout하며 z_task, z_mode 수집
    ├── classify_trajectories.py # CW/CCW 자동 분류
    ├── visualize.py             # UMAP + HDBSCAN 시각화
    └── run_analysis.py          # 분석 엔트리포인트 (argparse)
```

---

## 5. 클래스 인터페이스 (이대로 구현할 것)

### `modular_network.py`

```python
class ModularLayer(nn.Module):
    """단일 layer의 modular 파라미터 세트"""
    def __init__(self, in_features: int, out_features: int,
                 M_task: int, M_mode: int):
        # self.weights shape: (M_task * M_mode, out_features, in_features)
        # self.biases  shape: (M_task * M_mode, out_features)
        ...

    def forward(self, x: Tensor, w_combined: Tensor) -> Tensor:
        # x:          (B, in_features)
        # w_combined: (B, M_task * M_mode)
        # W_eff:      (B, out_features, in_features)  ← einsum으로 계산
        # return:     (B, out_features)
        ...


class ModularBaseNetwork(nn.Module):
    """3-layer modular MLP: 12→64→64→2"""
    def __init__(self, state_dim: int = 12, hidden_dim: int = 64,
                 action_dim: int = 2, M_task: int = 4, M_mode: int = 2,
                 residual_scale: float = 0.1):
        # layers: [ModularLayer(12,64), ModularLayer(64,64), ModularLayer(64,2)]
        ...

    def forward(self, state: Tensor, w_combined: Tensor) -> Tensor:
        # return: delta_a (B, 2), tanh * residual_scale 적용 완료
        ...
```

### `hierarchical_modulation.py`

```python
class HighLevelModulation(nn.Module):
    """Task 단계 감지 (느린 신호)"""
    def __init__(self, M_task: int = 4):
        # MLP: 12→64→32→16 (z_task)
        # head: Linear(16, M_task) → softmax (w_task)
        ...

    def forward(self, state: Tensor):
        # state (B, 12)에서 슬라이싱:
        #   block_pose    = state[:, 2:7]
        #   goal_pose     = state[:, 7:12]
        #   block_to_goal = state[:, 7:9] - state[:, 2:4]
        # input = cat([block_pose, goal_pose, block_to_goal], dim=-1)  → (B,12)
        # return: z_task (B, 16), w_task (B, M_task)
        ...


class LowLevelModulation(nn.Module):
    """Mode(CW/CCW) 감지 (빠른 신호)"""
    def __init__(self, M_mode: int = 2, history_len: int = 5):
        # input_dim = 2 + 2 + history_len*2 = 4 + history_len*2
        # MLP: input_dim→64→32→16 (z_mode)
        # head: Linear(16, M_mode) → softmax (w_mode)
        ...

    def forward(self, state: Tensor, traj_history: Tensor):
        # state (B, 12):
        #   pusher_xy     = state[:, 0:2]
        #   block_xy      = state[:, 2:4]
        # pusher_vel: traj_history에서 마지막 두 점의 차이로 계산
        # pusher_to_block = block_xy - pusher_xy
        # traj_flat = traj_history.view(B, -1)   → (B, history_len*2)
        # input = cat([pusher_vel, pusher_to_block, traj_flat], dim=-1)
        # return: z_mode (B, 16), w_mode (B, M_mode)
        ...


class HierarchicalModulation(nn.Module):
    """High + Low → w_combined"""
    def __init__(self, M_task: int = 4, M_mode: int = 2,
                 history_len: int = 5):
        ...

    def forward(self, state: Tensor, traj_history: Tensor):
        # w_combined = outer_product(w_task, w_mode).flatten()
        # 구현:
        #   outer = torch.bmm(w_task.unsqueeze(2), w_mode.unsqueeze(1))
        #   w_combined = outer.view(B, -1)   → (B, M_task*M_mode)
        # return: z_task, z_mode, w_combined, w_task, w_mode
        # (분석용으로 모두 반환)
        ...
```

### `modular_residual_policy.py`

```python
class ModularResidualPolicy(nn.Module):
    def __init__(self, diffusion_policy, M_task: int = 4, M_mode: int = 2,
                 history_len: int = 5, residual_scale: float = 0.1):
        # diffusion freeze:
        self.diffusion = diffusion_policy
        self.diffusion.requires_grad_(False)
        # 반드시 확인:
        assert not any(p.requires_grad for p in self.diffusion.parameters())
        ...

    def predict_action(self, obs_dict: dict,
                       traj_history: Tensor) -> dict:
        """
        return dict:
          'action'     : a_final    (B, Ta, 2)  ← 실제 실행할 action chunk
          'action_diff': a_diff     (B, Ta, 2)  ← diffusion 원본
          'delta_a'    : delta_a    (B, 2)       ← 현재 스텝 residual
          'z_task'     : (B, 16)   detach
          'z_mode'     : (B, 16)   detach
          'w_task'     : (B, M_task) detach
          'w_mode'     : (B, M_mode) detach
          'w_combined' : (B, M_task*M_mode) detach
        """
        # delta_a는 chunk의 모든 스텝에 동일하게 더함
        # (매 스텝 호출 시 현재 state로 새로 계산되므로 closed-loop)
        ...

    def get_trainable_parameters(self):
        # diffusion 제외. optimizer에 이것만 넣을 것.
        return (list(self.modulation.parameters()) +
                list(self.base_network.parameters()))
```

### `ppo_trainer.py`

```python
class RolloutBuffer:
    """PPO rollout 저장. z_task, z_mode도 함께 저장."""
    fields = ['state', 'action', 'reward', 'done', 'value',
              'log_prob', 'z_task', 'z_mode', 'w_task', 'w_mode']
    ...


class PPOTrainer:
    """
    하이퍼파라미터:
      gamma=0.99, gae_lambda=0.95, clip_eps=0.2
      n_steps=2048, batch_size=256, n_epochs=10
      vf_coef=0.5, ent_coef=0.01

    reward:
      r = (curr_coverage - prev_coverage)   # dense
          + 0.1 * float(curr_coverage > 0.95)  # sparse success bonus

    로깅 (wandb 또는 tensorboard):
      reward/mean_episode_reward
      reward/success_rate
      train/w_task_entropy    ← 감소하면 task 모듈 분화 중
      train/w_mode_entropy    ← 감소하면 mode 모듈 분화 중
      train/policy_loss
      train/value_loss
    """
    def __init__(self, policy: ModularResidualPolicy, env, config: dict):
        # optimizer: 파라미터 그룹별 lr 분리
        self.optimizer = torch.optim.Adam([
            {'params': policy.modulation.high_level.parameters(), 'lr': 1e-4},
            {'params': policy.modulation.low_level.parameters(),  'lr': 3e-4},
            {'params': policy.base_network.parameters(),           'lr': 1e-4},
        ])
        ...

    def train(self, total_timesteps: int):
        # 매 rollout마다 w_task_entropy, w_mode_entropy 로깅
        # checkpoint 저장 시 embedding buffer도 함께 저장
        ...
```

### `analysis/collect_embeddings.py`

```python
def collect_rollout_embeddings(policy: ModularResidualPolicy,
                                env, n_episodes: int = 200) -> dict:
    """
    return:
      'z_task'     : (N, 16)    N = 전체 스텝 수
      'z_mode'     : (N, 16)
      'w_task'     : (N, M_task)
      'w_mode'     : (N, M_mode)
      'episode_id' : (N,)
      'step_id'    : (N,)
      'pusher_xy'  : (N, 2)     CW/CCW 분류용
      'block_xy'   : (N, 2)     CW/CCW 분류용
      'success'    : (n_episodes,)  에피소드별 성공 여부
    """
    ...
```

### `analysis/classify_trajectories.py`

```python
def classify_cw_ccw(pusher_xy: np.ndarray,
                    block_xy: np.ndarray) -> str:
    """
    pusher의 block 중심 기준 angular displacement 누적값 부호로 판별.
    return: 'CW', 'CCW', 'ambiguous'
    """
    # block_xy를 원점으로, pusher 각도 변화 누적
    # 누적값 > threshold → CCW, < -threshold → CW
    ...

def label_episodes(embeddings: dict) -> dict:
    """
    embeddings에 'trajectory_label' (N,) 추가.
    각 스텝에 해당 에피소드의 CW/CCW 레이블 부여.
    레이블은 분석에만 사용, 학습에 사용 금지.
    """
    ...
```

### `analysis/visualize.py`

```python
def plot_umap_embeddings(embeddings: dict, save_dir: str):
    """
    저장 파일:
      umap_z_task.png  : z_task UMAP, CW/CCW 색상
      umap_z_mode.png  : z_mode UMAP, CW/CCW 색상
      umap_combined.png: 두 그림 나란히
    HDBSCAN 클러스터 경계 overlay.
    HAMNET Fig.13 형식 참고.
    """
    ...

def plot_episode_embedding_trajectory(embeddings: dict,
                                       episode_id: int,
                                       save_dir: str):
    """
    특정 에피소드의 z_task, z_mode 변화를 시간축으로.
    HAMNET Fig.14 형식 참고.
    """
    ...

def plot_module_activation_heatmap(embeddings: dict, save_dir: str):
    """
    CW vs CCW 에피소드에서 w_task, w_mode activation 비교 heatmap.
    """
    ...
```

---

## 6. 구현 순서 (반드시 이 순서대로)

### Step 1: 환경 확인
```bash
python eval.py \
  --checkpoint data/epoch=0550-test_mean_score=0.969.ckpt \
  --output_dir data/pusht_eval_output \
  --device cuda:0
```
확인: rollout 영상에서 CW/CCW 두 경로가 모두 나타나는가?

---

### Step 2: 네트워크 단위 테스트
구현 후 즉시 실행:
```python
# python -m diffusion_policy.diffmod.test_network
import torch
from diffusion_policy.diffmod.modular_network import ModularBaseNetwork
from diffusion_policy.diffmod.hierarchical_modulation import HierarchicalModulation

B = 4
state = torch.randn(B, 12)
traj_history = torch.randn(B, 5, 2)

mod = HierarchicalModulation(M_task=4, M_mode=2)
z_task, z_mode, w_combined, w_task, w_mode = mod(state, traj_history)

assert w_combined.shape == (B, 8)
assert torch.allclose(w_task.sum(-1), torch.ones(B), atol=1e-5)
assert torch.allclose(w_mode.sum(-1), torch.ones(B), atol=1e-5)

net = ModularBaseNetwork(M_task=4, M_mode=2, residual_scale=0.1)
delta_a = net(state, w_combined)
assert delta_a.shape == (B, 2)
assert (delta_a.abs() <= 0.1 + 1e-5).all()

print("PASS: 모든 shape/constraint 확인")
```

---

### Step 3: Freeze 확인
```python
# 반드시 확인
from diffusion_policy.diffmod.modular_residual_policy import ModularResidualPolicy

policy = ModularResidualPolicy(diffusion_policy)
assert not any(p.requires_grad for p in policy.diffusion.parameters()), \
    "FAIL: Diffusion이 freeze되지 않았음"

n_train = sum(p.numel() for p in policy.get_trainable_parameters())
n_total = sum(p.numel() for p in policy.parameters())
print(f"학습 파라미터: {n_train:,} / 전체: {n_total:,}")
# n_train이 n_total의 1~5% 수준이어야 정상
```

---

### Step 4: Flat M=2 먼저 (계층 없이)
계층 구조 전에 flat modular (M=2)가 작동하는지 확인.
`HierarchicalModulation` 대신 `FlatModulation(M=2)` 임시 구현.
100 에피소드 rollout 후 UMAP에서 2개 클러스터 확인.

---

### Step 5: PPO 학습
```bash
python -m diffusion_policy.diffmod.train_diffmod \
  --checkpoint data/epoch=0550-test_mean_score=0.969.ckpt \
  --output_dir data/diffmod_M4x2 \
  --M_task 4 --M_mode 2 \
  --total_timesteps 1_000_000 \
  --device cuda:0
```

모니터링:
- `w_task_entropy`, `w_mode_entropy` 감소 추세 확인
- `success_rate` ≥ 0.969 (base policy 이상 유지)

---

### Step 6: 분석
```bash
python -m diffusion_policy.diffmod.analysis.run_analysis \
  --checkpoint data/diffmod_M4x2/best.ckpt \
  --n_episodes 200 \
  --output_dir data/diffmod_analysis
```

---

## 7. 함정 포인트 (반드시 확인)

### [CRITICAL] Diffusion freeze
```python
self.diffusion.requires_grad_(False)
# optimizer 생성 시 반드시 get_trainable_parameters()만 사용
optimizer = Adam(policy.get_trainable_parameters())
```

### [CRITICAL] z_* detach
```python
# 분석용 반환값은 반드시 detach
# PPO loss 계산 시에는 detach 전 w_combined 사용
return {
    'z_task': z_task.detach().cpu(),
    'z_mode': z_mode.detach().cpu(),
    ...
}
```

### [CRITICAL] outer product 구현
```python
# w_task: (B, M_task=4), w_mode: (B, M_mode=2)
outer = torch.bmm(
    w_task.unsqueeze(2),    # (B, 4, 1)
    w_mode.unsqueeze(1)     # (B, 1, 2)
)                            # (B, 4, 2)
w_combined = outer.view(B, -1)  # (B, 8)
```

### [CRITICAL] Learning rate 분리
```python
# high-level은 느리게, low-level은 빠르게
optimizer = Adam([
    {'params': policy.modulation.high_level.parameters(), 'lr': 1e-4},
    {'params': policy.modulation.low_level.parameters(),  'lr': 3e-4},
    {'params': policy.base_network.parameters(),           'lr': 1e-4},
])
```

### [IMPORTANT] traj_history FIFO 관리
```python
from collections import deque
traj_history = deque(maxlen=5)
# 에피소드 시작 시: zeros로 초기화
for _ in range(5):
    traj_history.append(np.zeros(2))
# 매 스텝:
traj_history.append(current_pusher_xy.copy())
# network 입력:
history_tensor = torch.FloatTensor(list(traj_history))  # (5, 2)
```

### [IMPORTANT] action chunk 구조
```python
# diffusion은 Ta=16 스텝 chunk 생성
# 매 step_in_chunk마다: a_t = a_diff[step_in_chunk] + delta_a_t
# delta_a는 매 스텝 현재 state로 새로 계산 (closed-loop 보장)
# chunk는 Ta 스텝마다 재생성
```

### [IMPORTANT] CW/CCW 레이블 사용 제한
```python
# 레이블은 analysis/classify_trajectories.py에서만 생성
# 학습 코드(ppo_trainer.py) 어디에도 들어가면 안 됨
# 분석에서 사후 검증용으로만 사용
```

---

## 8. Ablation 계획

| 실험 | M_task | M_mode | 구조 | 목적 |
|------|--------|--------|------|------|
| A | 1 | 1 | flat | ResiP baseline |
| B | 1 | 2 | flat | mode만 분리 효과 |
| C | 2 | 1 | flat | task만 분리 효과 |
| D | 4 | 1 | flat | task 4개 분리 |
| **E** | **4** | **2** | **hierarchical** | **메인 모델** |
| F | 4 | 2 | flat (non-hier) | 계층 구조 효과 |

A→B: mode 분리의 효과
A→C/D: task 분리의 효과
E→F: 계층 구조 자체의 효과
