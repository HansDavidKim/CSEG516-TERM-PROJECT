# 🐳 Docker 사용 가이드

RLB-MI 공격을 Docker 환경에서 실행하는 방법입니다.

## 📋 사전 요구사항

### CPU 버전
- Docker 설치
- 최소 8GB RAM 권장

### GPU 버전
- Docker 설치
- NVIDIA GPU
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 설치
- CUDA 호환 GPU

## 🚀 빠른 시작

### 방법 1: Docker Compose 사용 (권장)

#### CPU 버전
```bash
# 빌드 및 실행
docker-compose --profile cpu up -d rlb-mi-cpu

# 컨테이너 접속
docker-compose exec rlb-mi-cpu bash

# 또는 한 번에
docker-compose --profile cpu run --rm rlb-mi-cpu bash
```

#### GPU 버전
```bash
# 빌드 및 실행
docker-compose --profile gpu up -d rlb-mi-gpu

# 컨테이너 접속
docker-compose exec rlb-mi-gpu bash

# 또는 한 번에
docker-compose --profile gpu run --rm rlb-mi-gpu bash
```

### 방법 2: Docker 명령어 직접 사용

#### CPU 버전
```bash
# 이미지 빌드
docker build -t rlb-mi:cpu -f Dockerfile .

# 컨테이너 실행 (인터랙티브)
docker run -it --rm \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/attack_results:/app/attack_results \
  -v $(pwd)/dataset:/app/dataset \
  rlb-mi:cpu bash
```

#### GPU 버전
```bash
# 이미지 빌드
docker build -t rlb-mi:gpu -f Dockerfile.gpu .

# 컨테이너 실행 (GPU 사용)
docker run -it --rm \
  --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/attack_results:/app/attack_results \
  -v $(pwd)/dataset:/app/dataset \
  rlb-mi:gpu bash
```

## 📚 사용 예시

### 1. 데이터 준비 (호스트에서)

```bash
# 체크포인트 디렉토리 생성
mkdir -p checkpoints attack_results dataset pretrained

# 필요한 파일들 배치
# - checkpoints/generator_last.pt
# - checkpoints/vgg16_celeba_best.pt
# - dataset/... (옵션)
```

### 1.5. 데이터 로드 및 모델 학습 (컨테이너 내부)

```bash
# 1. 데이터 로드 (Kaggle 다운로드 및 전처리)
# .env 파일에 KAGGLE_USERNAME, KAGGLE_KEY가 설정되어 있어야 합니다.
python main.py load-data

# 2. Classifier 학습 (VGG16, CelebA)
python main.py train-classifier \
  --data-set celeba \
  --model-name VGG16 \
  --epoch 50 \
  --batch-size 64

# 3. Generator 학습 (FFHQ)
python main.py train-generator \
  --data-root dataset/public/flickrfaceshq-dataset-ffhq \
  --epochs 50 \
  --batch-size 128 \
  --output-dir checkpoints
```

### 2. 컨테이너 내부에서 공격 실행

```bash
# 컨테이너 접속 후
python main.py run-rlb-mi-attack \
  --generator checkpoints/generator_last.pt \
  --target-model checkpoints/vgg16_celeba_best.pt \
  --model-name VGG16 \
  --target-class 0 \
  --num-classes 1000 \
  --episodes 40000 \
  --alpha 0.0 \
  --num-images 1000 \
  --top-k 10 \
  --output-dir attack_results
```

### 3. 예제 스크립트 실행

```bash
# 컨테이너 내부에서
python example_rlb_mi.py
```

## 🔧 고급 사용법

### 특정 GPU 선택

```bash
# GPU 0번만 사용
docker run -it --rm \
  --gpus '"device=0"' \
  -v $(pwd)/checkpoints:/app/checkpoints \
  rlb-mi:gpu bash

# GPU 0, 1번 사용
docker run -it --rm \
  --gpus '"device=0,1"' \
  -v $(pwd)/checkpoints:/app/checkpoints \
  rlb-mi:gpu bash
```

### 백그라운드에서 실행

```bash
# 컨테이너를 백그라운드로 실행
docker run -d \
  --name rlb-mi-attack \
  --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/attack_results:/app/attack_results \
  rlb-mi:gpu \
  python main.py run-rlb-mi-attack \
    --generator checkpoints/generator_last.pt \
    --target-model checkpoints/vgg16_celeba_best.pt \
    --target-class 0 \
    --episodes 40000

# 로그 확인
docker logs -f rlb-mi-attack

# 종료 후 결과 확인
ls -la attack_results/
```

### 환경 변수 설정

```bash
docker run -it --rm \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONUNBUFFERED=1 \
  --gpus all \
  rlb-mi:gpu bash
```

## 📊 볼륨 마운트 설명

| 호스트 경로 | 컨테이너 경로 | 설명 |
|----------|------------|------|
| `./checkpoints` | `/app/checkpoints` | 학습된 모델 체크포인트 |
| `./attack_results` | `/app/attack_results` | 공격 결과 (이미지, agent) |
| `./dataset` | `/app/dataset` | 데이터셋 (선택사항) |
| `./pretrained` | `/app/pretrained` | 사전학습 모델 (선택사항) |

## 🛠️ 트러블슈팅

### GPU가 인식되지 않을 때

```bash
# NVIDIA Container Toolkit 설치 확인
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# 위 명령이 실패하면 NVIDIA Container Toolkit 재설치 필요
```

### 메모리 부족 에러

```bash
# Docker 메모리 제한 늘리기 (16GB 예시)
docker run -it --rm \
  --memory=16g \
  --gpus all \
  rlb-mi:gpu bash
```

### 권한 문제

```bash
# 호스트와 동일한 UID/GID로 실행
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v $(pwd)/checkpoints:/app/checkpoints \
  rlb-mi:cpu bash
```

## 🧹 정리

### 컨테이너 중지 및 삭제

```bash
# Docker Compose 사용 시
docker-compose --profile cpu down
docker-compose --profile gpu down

# 직접 실행 시
docker stop rlb-mi-attack
docker rm rlb-mi-attack
```

### 이미지 삭제

```bash
docker rmi rlb-mi:cpu
docker rmi rlb-mi:gpu
```

### 전체 정리 (주의: 모든 Docker 리소스 삭제)

```bash
# 사용하지 않는 컨테이너, 이미지, 볼륨 삭제
docker system prune -a
```

## 📝 유용한 명령어

```bash
# 실행 중인 컨테이너 확인
docker ps

# 모든 컨테이너 확인
docker ps -a

# 이미지 목록 확인
docker images

# 컨테이너 로그 확인
docker logs <container_name>

# 컨테이너 내부 파일 복사
docker cp <container_name>:/app/attack_results ./local_results

# 리소스 사용량 확인
docker stats
```

## 🎯 완전한 예시 워크플로우

```bash
# 1. 이미지 빌드
docker-compose --profile gpu build

# 2. 컨테이너 실행 및 접속
docker-compose --profile gpu run --rm rlb-mi-gpu bash

# 3. 컨테이너 내부에서 공격 실행
python main.py run-rlb-mi-attack \
  --generator checkpoints/generator_last.pt \
  --target-model checkpoints/vgg16_celeba_best.pt \
  --target-class 0 \
  --episodes 40000

# 4. 결과 확인 (호스트에서)
# exit로 컨테이너 종료 후
ls -la attack_results/
```

## 🔗 참고 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Compose 문서](https://docs.docker.com/compose/)

## 🔐 Kaggle 인증 설정

Kaggle 데이터셋을 사용하려면 Kaggle API 자격증명이 필요합니다.

### 방법 1: .env 파일 사용 (권장)

1. 프로젝트 루트에 `.env` 파일 생성:
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
2. `docker-compose.yml`에서 환경 변수로 자동 로드됩니다.

### 방법 2: kaggle.json 파일 사용

# 호스트에서 kaggle.json 준비
# Windows
mkdir $env:USERPROFILE\.kaggle
Copy-Item kaggle.json $env:USERPROFILE\.kaggle\kaggle.json

# Linux/Mac
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json그리고 `docker-compose.yml`에 볼륨 추가:aml
volumes:
  - ~/.kaggle:/root/.kaggle## 📊 데이터 로드 및 학습 워크플로우

### 1. 데이터 로드

# 컨테이너 접속
docker-compose --profile gpu run --rm rlb-mi-gpu bash

# 컨테이너 내부에서 데이터 로드
python main.py load-data이 명령은:
- Kaggle에서 데이터셋 다운로드
- 데이터 전처리 및 public/private split 생성
- `dataset/` 디렉토리에 저장

### 2. Classifier 학습

# 컨테이너 내부에서
python main.py train-classifier \
  --data-set celeba \
  --model-name VGG16 \
  --epoch 50 \
  --batch-size 64학습된 모델은 `checkpoints/vgg16_celeba_best.pt`에 저장됩니다.

### 3. Generator 학습
ash
# Public 데이터셋으로 Generator 학습
python main.py train-generator \
  --data-root dataset/public/flickrfaceshq-dataset-ffhq \
  --epochs 50 \
  --batch-size 128 \
  --output-dir checkpoints### 4. 완전한 워크플로우 예시
sh
# 1. 컨테이너 실행 및 접속
docker-compose --profile gpu run --rm rlb-mi-gpu bash

# 2. 데이터 로드
python main.py load-data

# 3. Classifier 학습
python main.py train-classifier --data-set celeba --model-name VGG16

# 4. Generator 학습
python main.py train-generator --data-root dataset/public/flickrfaceshq-dataset-ffhq

# 5. 공격 실행 (선택사항)
python main.py run-rlb-mi-attack \
  --generator checkpoints/generator_last.pt \
  --target-model checkpoints/vgg16_celeba_best.pt \
  --target-class 0### 백그라운드에서 학습 실행

# 컨테이너를 백그라운드로 실행하여 학습
docker-compose --profile gpu run -d \
  --name rlb-mi-training \
  rlb-mi-gpu \
  python main.py train-classifier --data-set celeba --model-name VGG16

# 로그 확인
docker logs -f rlb-mi-training

# 학습 완료 후 결과 확인
ls -la checkpoints/## 🗂️ 디렉토리 구조

학습 후 생성되는 디렉토리 구조:

```
.
├── dataset/
│   ├── private/
│   │   ├── celeba/          # Private 데이터셋
│   │   ├── facescrub-full/
│   │   └── pubfig83/
│   └── public/
│       └── flickrfaceshq-dataset-ffhq/  # Public 데이터셋
├── checkpoints/
│   ├── vgg16_celeba_best.pt  # 학습된 Classifier
│   └── generator_last.pt     # 학습된 Generator
└── attack_results/            # 공격 결과
```

## ⚠️ 주의사항

1. **데이터 크기**: Kaggle 데이터셋은 크기가 클 수 있으므로 충분한 디스크 공간을 확보하세요.
2. **학습 시간**: GPU 사용을 권장합니다. CPU로는 학습 시간이 매우 오래 걸릴 수 있습니다.
3. **메모리**: 대용량 데이터셋의 경우 Docker 메모리 제한을 늘려야 할 수 있습니다.
