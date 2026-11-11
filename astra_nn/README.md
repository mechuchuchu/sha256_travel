# SHA-256: 512비트 입력 → 256비트 출력 신경망

SHA-256 연산을 **고정 가중치의 affine → ReLU → affine 게이트**로 구현한
PyTorch 모델입니다. 입력이 정확히 `0` 또는 `1`이면 결과도 정확한 비트입니다.
학습한 근사 모델이 아니라, SHA-256 회로를 직접 구성한 신경망입니다.

| 파일 | 내용 |
| --- | --- |
| `sha256_nn.py` | 모델 구조, 가중치 생성·저장·불러오기, 비트 변환 함수 |
| `sha256_512_256.pt` | 원본 64바이트 메시지용 체크포인트. 패딩 포함 두 블록 처리 |
| `sha256_compression_512_256.pt` | 이미 준비된 512비트 블록용 체크포인트. 표준 IV로 한 블록 처리 |
| `sha256_weights.json` | 모든 게이트 가중치·편향과 상수를 읽을 수 있는 JSON |
| `test_sha256_nn.py` | `hashlib` 대조, carry 전파, 모든 입력 비트 위치 등 검증 |

512비트 **원본 메시지**에는 별도의 패딩 블록이 추가되므로 압축을 두 번 합니다.
이미 패딩된 한 블록을 넣는 경우에는 `compression` 모델을 사용합니다.
패딩과 압축 구조는 [NIST FIPS 180-4](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)의
5.1.1절과 6.2절을 따릅니다.

## 실행

이 디렉터리에서 실행합니다. 추가 의존성은 PyTorch뿐입니다.

```bash
uv run python sha256_nn.py --export .
uv run python -m unittest -v test_sha256_nn.py
```

원본 64바이트 메시지:

```python
import hashlib
import torch
from sha256_nn import SHA256NN, bytes_to_bits, bits_to_bytes

model = SHA256NN.load("sha256_512_256.pt")
message = bytes(range(64))
x = bytes_to_bits(message)             # shape: (512,), float32, 0/1
with torch.inference_mode():
    y = model(x)                       # shape: (256,), 정확한 0/1
digest = bits_to_bytes(y)
print(digest.hex())
assert digest == hashlib.sha256(message).digest()
```

출력:

```text
fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108
```

이미 패딩된 512비트 블록:

```python
import hashlib
import torch
from sha256_nn import SHA256NN, bytes_to_bits, bits_to_bytes, pad_single_block

model = SHA256NN.load("sha256_compression_512_256.pt")
block = pad_single_block(b"abc")        # 0..55바이트 메시지만 한 블록에 들어감
with torch.inference_mode():
    y = model(bytes_to_bits(block))
assert bits_to_bytes(y) == hashlib.sha256(b"abc").digest()
```

입력·출력 형태는 `(..., 512)`와 `(..., 256)`이므로 배치도 지원합니다.
외부 비트 순서는 바이트 순서 그대로, 각 바이트의 **MSB부터**입니다.
상위 프로젝트의 기존 배열은 32비트 워드 안에서 LSB부터 저장하므로 변환이 필요합니다:

```python
# 기존 입력: (batch, 512), 워드 내부 LSB 우선
x_msb = x_old.reshape(-1, 16, 32).flip(-1).reshape(-1, 512)
y_msb = model(x_msb)
y_old = y_msb.reshape(-1, 8, 32).flip(-1).reshape(-1, 256)
```

## 가중치와 구조

각 게이트는 `y = W_out @ ReLU(W_in @ x + b_in) + b_out`입니다.
모든 가중치는 작은 정수이며 float32로 저장됩니다. 게이트 가중치와 편향은
총 **73개 스칼라**이고, 비트 위치와 라운드마다 공유됩니다. 학습은 필요하지 않습니다.
SHA 초기값·라운드 상수·패딩은 체크포인트의 별도 버퍼에 포함됩니다.

예를 들어 XOR와 AND를 함께 계산하는 게이트는 다음과 같습니다.

```text
W_in  = [[1, 1],     b_in  = [0, -1]
         [1, 1]]

W_out = [[1, -2],    b_out = [0, 0]
         [0,  1]]
```

첫 출력은 `x XOR y`, 두 번째 출력은 `x AND y`입니다.
carry 전파는 다섯 단계의 prefix 회로로 구현합니다. XOR3, Ch, Maj도 같은
형태의 ReLU 게이트입니다. 회전·이동·비트 재배치는 고정 연결이며, 입력에 대한
정수 비트 연산이나 `hashlib` 호출은 모델의 forward에 들어가지 않습니다.

73개는 **공유되는 가중치 수**입니다. 전체 연산 그래프는 SHA의 64라운드를
블록마다 수행하므로 작거나 빠른 범용 MLP를 학습했다는 의미는 아닙니다.
체크포인트는 이 코드의 연결 구조와 함께 사용해야 합니다.

정확성은 제공된 가중치와 이진 입력에 대해 성립합니다. `forward`는 소수 입력을
거부합니다. ReLU 경계의 입력 미분이 해시 역산에 유용하다는 보장은 없습니다.
float32·float64 CPU 실행을 테스트하며, GPU 및 저정밀 실행은 별도 검증 대상입니다.

실수 입력이 회로를 어떻게 통과하는지 확인하려면 `FractionalSHA256NN`을 사용합니다.
이 버전은 유한한 `[0, 1]` 값을 그대로 받지만, 소수 입력의 출력은 이진 해시가 아닙니다.

```python
import torch
from sha256_nn_fractional import FractionalSHA256NN

x = torch.zeros(512)
x[0] = 0.5
with torch.inference_mode():
    y = FractionalSHA256NN()(x)
print(y[:8])
```
