# HAMNet 구조 분석: Modular RL + 병렬 학습

> 참고 코드: `referance/HAMNet/ham/src/ham/`
> UNICORN (포인트 클라우드 인코더)는 제외하고 분석

---

## 1. 전체 아키텍처 개요

```
train.py
  ├── load_env()   → PushEnv (IsaacGym, num_env개 병렬) + ObservationWrapper 체인
  └── load_agent() → MLPStateEncoder + PiNet/VNet + PPO
```

최종 제어 흐름:
```
observation (num_env, obs_dim)
    → MLPStateEncoder    : GRU 기반 상태 집계
    → ModularHyperNet    : 변조 네트워크 → 모듈 가중치 → GateModMLP
    → action (num_env, action_dim)
```

---

## 2. Modular RL 핵심 구조

### 2-1. 핵심 클래스

| 파일 | 클래스 | 역할 |
|------|--------|------|
| `models/rl/net/hypernet_modular.py` | `ModularHyperActionValueNet` | 전체 Modular 정책+가치 네트워크 |
| `models/rl/net/hypernet_modular.py` | `NodeModulationNetwork` | 계층별 모듈 선택 가중치 생성 |
| `models/rl/net/hypernet_modular.py` | `EdgeModulationNetwork` | 레이어 간 자동회귀 모듈 연결 |
| `models/rl/net/hypernet_modular.py` | `ActionValueSubnet` | 개별 서브넷 기본 구조 |
| `models/rl/net/gate_mod_mlp.py` | `GateModMLP` | 모듈 혼합 실행 (einsum 기반) |

### 2-2. Modular Hypernetwork 구조

```
입력: state (num_env, state_dim), context (optional)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│          변조 네트워크 (Modulation Network)          │
│                                                     │
│  NodeModulationNetwork:                             │
│    Linear → Softmax(temperature)                    │
│    출력: weights (num_env, num_layers, num_module)  │
│                                                     │
│  (또는 EdgeModulationNetwork: AR 방식)              │
└──────────────────┬──────────────────────────────────┘
                   │  weights (모듈 선택 확률)
                   ▼
┌─────────────────────────────────────────────────────┐
│              GateModMLP (모듈 혼합 실행)             │
│                                                     │
│  각 레이어:                                          │
│    Wx = einsum('bi,bm,mij,bj->bi', gW, pW, Ws, x)  │
│    b  = einsum('bj,bm,mj->bj',    gb, pb, bs)      │
│    output = act(norm(Wx + b))                       │
│                                                     │
│  - Ws: (num_module, out_dim, in_dim) 각 모듈 가중치 │
│  - pW: (num_env, num_module) 모듈 선택 확률         │
│  - gW: 게이팅 스케일 (선택적)                       │
└──────────────────┬──────────────────────────────────┘
                   │
         (action_mean, log_std, value)
```

### 2-3. 모듈 파라미터 관리 방식 (Functorch)

```python
# ActionValueSubnet을 num_module개 복사 → 파라미터 스택
params = nn.ParameterList(stacked_params)
# shape: (num_params, num_module, param_dims)

# make_functional_with_buffers 로 함수형 변환
func, buf = make_functional_with_buffers(subnet)

# 배치 forward: 모든 모듈을 동시에 계산
output = vmap(func)(params, buf, x)
```

### 2-4. 주요 설정 파라미터 (`ModularHyperActionValueNet.Config`)

| 파라미터 | 의미 | 비고 |
|----------|------|------|
| `num_module` | 레이어당 모듈 수 | 클수록 표현력 ↑ |
| `auto_regressive` | AR 방식 엣지 모듈화 | EdgeModulationNetwork 활성화 |
| `scaling` / `fuse_scale` | 게이팅 스케일 활성화 | `NodeModulationNetworkWithGate` |
| `sm_like_modulation` | Switch-Mixture 스타일 | sparse 모듈 선택 |
| `temperature` | softmax 온도 | 낮을수록 hard selection |
| `key_state` / `dim_state` | 상태 입력 키/차원 | - |

---

## 3. 병렬 환경 학습 구조

### 3-1. IsaacGym 벡터화 환경

```
PushEnv (push_env.py)
  ├── num_env: 병렬 환경 수 (e.g., 64, 512)
  ├── 모든 텐서: GPU에 직접 할당
  │     obs: (num_env, obs_dim)
  │     rew: (num_env, num_rew)
  │     done: (num_env,)
  └── step(actions: Tensor[num_env, action_dim])
        → (obs, rew, done, info) 한 번에 반환
```

환경 생성 흐름 (`train.py: load_env()`):
```
make_arm_env(cfg.env)
    → PushEnv (IsaacGym, num_env 병렬)
    → AddPhysParams wrapper   (물체 물리 파라미터 추가)
    → AddPrevAction wrapper   (이전 액션 관찰에 포함)
    → RelGoal wrapper         (절대 → 상대 좌표 변환)
    → NormalizeEnv wrapper    (관찰/보상 정규화)
    → MonitorEnv wrapper      (성능 모니터링)
```

### 3-2. PPO 학습 루프

```
PPO.learn()
  │
  ├─ [수집 단계] interact() × rollout_size
  │    for t in range(rollout_size):
  │      actn = policy(state)              # (num_env, action_dim)
  │      obs, rew, done = env.step(actn)   # num_env 병렬
  │      buffer.append(obs, actn, rew, done)
  │
  └─ [학습 단계] _maybe_train()
       └─ _train_epoch() × epoch
            ├─ _build_batch()   : 상태 재계산 + GAE
            └─ _train_batch()   : PPO loss + optimizer.step()
```

### 3-3. DictBuffer: 순환 데이터 버퍼

```python
# 저장 형식
buffer['obsn']:   (rollout_size, num_env, obs_dim)
buffer['actn']:   (rollout_size, num_env, action_dim)
buffer['logp']:   (rollout_size, num_env)
buffer['rewd']:   (rollout_size, num_env, num_rew)
buffer['done']:   (rollout_size, num_env)
buffer['hidden']: (rollout_size, num_env, state_dim)
```

### 3-4. 배치 구성 (`_build_batch`)

```
버퍼 데이터 (rollout_size, num_env, ...)
    ├─ _compute_state()  : RNN으로 상태 시퀀스 재계산 (BPTT)
    └─ _compute_targets(): GAE로 advantage/return 계산
         │
         ▼
배치 (chunk_size, td_horizon, num_env, ...)
  chunk_size × td_horizon = rollout_size
  e.g., 32 × 16 = 512
```

### 3-5. Gradient Accumulation

```python
mini_batch_size = num_env // accumulate_gradient
# e.g., 64 // 4 = 16

for j in range(P):            # P = accumulate_gradient
    D_j = split_by_env(D, j)  # 환경 차원으로 미니배치 분할
    loss = compute_loss(D_j) / P
    loss.backward()            # gradient 누적

optimizer.step()               # P개 미니배치 합산 후 업데이트
```

### 3-6. BPTT (Backpropagation Through Time) 설정

| 파라미터 | 기본값 | 의미 |
|----------|--------|------|
| `bptt_seq_len` | 8 | 역전파 길이 |
| `bptt_burn_in` | 6 | warm-up 스텝 (gradient 없이 상태 초기화) |
| `bptt_stride` | 4 | 시간축 스트라이드 |

---

## 4. 주요 하이퍼파라미터 (`ppo_config.py`)

| 카테고리 | 파라미터 | 기본값 | 의미 |
|----------|----------|--------|------|
| **수집** | `rollout_size` | 512 | 배치 구성 전 수집 스텝 수 |
| | `chunk_size` | 32 | 배치 분할 수 |
| | `td_horizon` | 16 | 유효 롤아웃 길이 (=rollout/chunk) |
| **PPO** | `gamma` | 0.99 | 할인 인자 |
| | `lmbda` | 0.95 | GAE λ |
| | `clip` | 0.3 | PPO 클립 범위 |
| | `epoch` | 5 | 배치 재사용 에포크 |
| **학습** | `lr` | 3e-4 | 학습률 |
| | `train_steps` | 8192 | 총 학습 반복 수 |
| | `accumulate_gradient` | 1 | gradient 누적 횟수 |

---

## 5. 상태 인코더: MLPStateEncoder

```
observation (num_env, obs_dim)
    │
    ├─ feature_encoders (각 관찰 키별 MLP)
    │    joint_pos   → MLP → feature_a
    │    joint_vel   → MLP → feature_b
    │    goal        → MLP → feature_c
    │    ...
    │
    ├─ feature_aggregators (GRU, 시계열 통합)
    │    hidden_prev × feature_a → GRU → new_feat_a
    │    ...
    │
    └─ feature_fuser (MLP, 특성 융합)
         [feat_a, feat_b, feat_c, ...] → concat → MLP → state
```

---

## 6. 파일 구조 요약

```
ham/src/ham/
├── models/rl/
│   ├── net/
│   │   ├── hypernet_modular.py  ← ModularHyperActionValueNet (핵심)
│   │   ├── gate_mod_mlp.py      ← GateModMLP (모듈 혼합 실행)
│   │   └── nets.py              ← PiNet, VNet (기본 정책/가치)
│   ├── v6/
│   │   └── ppo.py               ← PPO (interact, learn, _train_epoch)
│   ├── generic_state_encoder.py ← MLPStateEncoder
│   ├── ppo_config.py            ← DomainConfig, TrainConfig
│   ├── padded_buffer.py         ← DictBuffer (순환 버퍼)
│   ├── adaptive_lr.py           ← KL 기반 학습률 적응
│   └── env_normalizer.py        ← 관찰/보상 정규화
│
├── env/
│   ├── push_env.py              ← PushEnv (IsaacGym 벡터화)
│   ├── arm_env.py               ← ArmEnvWrapper, ArmEnvConfig
│   └── common.py                ← EnvBase, EnvIface
│
└── scripts/
    ├── train.py                 ← 진입점 (Config, load_env, load_agent)
    └── env_wrappers.py          ← ObservationWrapper 체인
```

---

## 7. diffmod 구현 시 참고 포인트

### UNICORN 제거 시 수정 사항
- `train.py`: `use_unicorn=False`, `AddObjectFullCloud` / `AddSceneFullClouds` 제거
- `ModularHyperActionValueNet.Config`: `key_ctx=None`, `dim_ctx=None` 으로 설정
- `get_network_modulation()`: `embed_context()` 단계 스킵 → 상태만으로 변조 가중치 계산

### Diffusion + Modular RL 결합 시 인터페이스
```
diffusion_action = diffusion_policy(obs)      # base action
residual_action  = modular_rl(obs, state)     # residual
final_action     = diffusion_action + residual_action
```
- Modular RL의 출력 범위 제한 필요 (잔차이므로 작은 값)
- `clip_action` 또는 `tanh` 스케일링 고려
- 보상: 성공/실패 sparse reward (Diffusion이 이미 목표 근방 제공)
