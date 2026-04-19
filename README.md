# Assignment 2: DeepXplore를 이용한 차등 테스팅

**과목:** Reliable and Trustworthy Artificial Intelligence  
**작성자:** Dohyun Kim  

---

## 개요

이 프로젝트는 신경망을 위한 화이트박스 차등 테스팅 프레임워크인 **DeepXplore**를 **CIFAR-10**과 **ResNet50** 환경에 맞게 PyTorch로 재구현합니다.

서로 다른 하이퍼파라미터와 랜덤 시드로 학습된 두 ResNet50 모델을 준비하고, DeepXplore를 통해 두 모델이 **서로 다른 예측을 내놓는 입력(disagreement input)**을 탐색하여 모델의 잠재적 취약점을 노출합니다.

> 참고 논문: Pei et al., "DeepXplore: Automated Whitebox Testing of Deep Learning Systems", SOSP 2017.

---

## 프로젝트 구조

```
.
├── train.py          # ResNet50 모델 2개 학습
├── deepxplore.py     # DeepXplore 핵심 로직 (PyTorch 재구현)
├── test.py           # DeepXplore 실행 및 결과 저장
├── requirements.txt  # Python 의존성 목록
├── results/          # 시각화 결과 및 요약
│   ├── disagreement_01.png
│   ├── ...
│   ├── disagreement_10.png
│   └── summary.txt
└── README.md
```

> **참고:** 모델 체크포인트(`models/`)와 CIFAR-10 데이터셋(`data/`)은 GitHub 파일 크기 제한으로 인해 저장소에 포함되지 않습니다. `python train.py`를 실행하면 재현할 수 있습니다.

---

## 환경 설정

### 1. 저장소 클론

```bash
git clone https://github.com/DohyunKim-UOS/TrustworthyAI_hw2.git
cd TrustworthyAI_hw2
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

> Python 3.9, CUDA 11.8 환경에서 테스트했습니다. CUDA GPU 사용을 권장합니다.

---

## 실행 방법

### Step 1: ResNet50 모델 2개 학습

```bash
python train.py
```

두 모델은 **서로 다른 하이퍼파라미터와 랜덤 시드**로 학습됩니다:

| | Model A | Model B |
|---|---|---|
| 시드 | 42 | 7 |
| 학습률 | 0.1 | 0.05 |
| 데이터 증강 | 기본 (Flip + Crop) | 강화 (+ ColorJitter + RandomErasing) |
| 에폭 | 50 | 50 |

학습 완료 후 체크포인트가 저장됩니다:
- `models/model_a_best.pth`
- `models/model_b_best.pth`

학습 시간: GPU 기준 모델당 약 30~60분 소요

---

### Step 2: DeepXplore 실행

```bash
python test.py
```

옵션을 직접 지정하려면:

```bash
python test.py --model_a models/model_a_best.pth \
               --model_b models/model_b_best.pth \
               --max_seeds 300 \
               --steps 150 \
               --epsilon 0.2
```

#### 실행 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model_a` | `models/model_a_best.pth` | Model A 체크포인트 경로 |
| `--model_b` | `models/model_b_best.pth` | Model B 체크포인트 경로 |
| `--max_seeds` | `200` | 테스트할 seed 이미지 수 |
| `--steps` | `100` | seed당 최대 perturbation 반복 횟수 |
| `--step_size` | `0.01` | gradient 업데이트 크기 |
| `--epsilon` | `0.3` | L-inf perturbation 최대 범위 |
| `--lambda_cov` | `0.5` | coverage loss 가중치 |
| `--save_dir` | `results` | 결과 저장 디렉토리 |

#### 출력 결과

- `results/disagreement_01.png` ~ `disagreement_10.png` — disagreement input 시각화 (원본 seed / 퍼터베이션 이미지 / 차이 ×10 비교)
- `results/summary.txt` — 전체 실행 요약 (뉴런 커버리지, 샘플별 예측 결과 포함)

---

## 실험 결과

`--max_seeds 300 --steps 150 --epsilon 0.2` 조건으로 실행한 결과:

| 지표 | 값 |
|---|---|
| 테스트한 seed 수 | 300 |
| Disagreement 발견 수 | **300개 (100%)** |
| 뉴런 커버리지 — Model A | **31.93%** |
| 뉴런 커버리지 — Model B | **52.63%** |

**주요 관찰:**

- **Disagreement rate 100%:** 모든 seed 이미지에서 두 모델이 다른 예측을 내놓는 입력을 찾을 수 있었습니다. 이는 두 모델이 서로 다른 학습 설정으로 인해 실질적으로 다른 decision boundary를 가지고 있음을 보여줍니다.
- **커버리지 차이 (A: 31.93% vs B: 52.63%):** 강한 데이터 증강(ColorJitter + RandomErasing)으로 학습된 Model B가 더 많은 뉴런을 활성화하며, 더 다양한 특징을 학습했음을 시사합니다.
- **커버리지 수렴 패턴:** 초반 seed에서 커버리지가 빠르게 증가하다가 후반부로 갈수록 증가폭이 줄어드는 전형적인 포화 패턴이 관찰됩니다.

---

## 원본 DeepXplore 대비 수정 사항

원본 DeepXplore(Pei et al.)는 Keras 기반으로 구현되어 있으며 ImageNet 등 대규모 데이터셋을 대상으로 합니다. 이 프로젝트에서는 CIFAR-10 + ResNet50 + PyTorch 환경에 맞게 다음과 같이 수정했습니다:

1. **PyTorch 완전 재구현** — Keras 포팅 대신 핵심 로직을 PyTorch로 처음부터 새로 구현했습니다.
2. **CIFAR-10용 ResNet50 구조 수정** — 32×32 입력에 맞게 첫 번째 conv를 7×7(stride 2) → 3×3(stride 1)으로 교체하고, maxpool을 제거하여 공간 해상도를 유지했습니다.
3. **Forward hook 기반 뉴런 커버리지** — 모델 구조를 변경하지 않고 `register_forward_hook`을 모든 ReLU 레이어에 등록하여 뉴런 활성화를 추적합니다. ResNet의 Bottleneck 블록에서 ReLU 인스턴스가 공유될 때 발생하는 키 충돌을 방지하기 위해 `id(module)`을 키에 포함했습니다.
4. **Joint loss 설계** — disagreement loss(두 모델의 예측을 벌리는 방향)와 coverage loss(평균 활성화 최대화)를 `lambda_cov` 가중치로 결합했습니다.
5. **FGSM 방식의 반복 perturbation** — 원본의 도메인별 변환 대신, L-inf projection을 적용한 반복적 gradient sign 업데이트(PGD 방식)를 사용했습니다.

## LLM 활용 명시
1. README.md 파일을 작성하는데 LLM을 활용하였음.
2. deepxplore.py 파일의 작성과 디버깅에 LLM을 활용하였음.
