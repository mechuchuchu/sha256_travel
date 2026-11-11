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

## mpmath 순전파·역전파

`sha256_nn_mpmath.py`의 `FractionalSHA256MP`는 같은 고정 ReLU 게이트,
5단계 carry-prefix 덧셈, 블록당 64라운드를 mpmath로 계산합니다.
게이트 가중치와 상수는 `sha256_weights.json`에서 읽습니다. 실행 의존성은
mpmath이며, PyTorch는 비교 테스트에만 사용합니다. 입력·출력은 각각
길이 512·256의 단일 벡터이고, 비트 순서는 기존 모델과 같은 MSB 우선입니다.

기본 정밀도는 **유효숫자 128비트**입니다. IEEE binary128(`float128`)의
유효숫자는 113비트이므로 그 정밀도를 원하면 `--precision 113`을 사용합니다.
mpmath의 지수 범위와 반올림 처리는 IEEE binary128을 완전히 모사하지 않습니다.
순전파 값과 역전파 누적값은 독립된 mpmath context에서 계산하며 전역 정밀도를
바꾸지 않습니다. ReLU의 0에서의 미분은 PyTorch와 동일하게 0을 선택합니다.

```bash
# 랜덤 fractional 입력 → SHA256(b"abc") 목표의 MSE와 입력 gradient
uv run python sha256_nn_mpmath.py --precision 128 --output fractional_mpmath_128_log.json

# abc의 패딩된 한 블록: 기본 목표가 abc의 해시이므로 MSE와 gradient가 0
uv run python sha256_nn_mpmath.py --mode compression --message-hex 616263 --output abc_mpmath.json

# 다른 목표 출력 및 기존 입력으로 평가
uv run python sha256_nn_mpmath.py --input-json fractional_rand_log.json \
  --target-hex ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad

uv run python -m unittest -v test_sha256_nn_mpmath.py
```

입력 자체를 업데이트하려면 mpmath 학습 실행기를 사용합니다. 아래 명령은
앞서 저장한 랜덤 fractional 입력을 시작점으로 삼아 `SHA256(b"abc")`의 256비트
출력에 맞춥니다. 입력은 매 단계 `[0, 1]`로 투영됩니다.

```bash
uv run python train_sha256_nn_mpmath.py \
  --precision 113 --input-json fractional_mpmath_128_log.json \
  --optimizer sign-line-search --steps 3 --lr 1e-16 --backtracks 4 \
  --output fractional_mpmath_train_113_linesearch.json
```

학습 기록에는 각 단계의 갱신 전 MSE와 gradient 크기, 학습 중 가장 낮은
MSE를 낸 fractional 입력·출력, 그리고 그 입력을 0.5 기준으로 이진화한
메시지의 실제 SHA-256이 들어갑니다.
기본 `sign-line-search`는 현재 gradient 부호 방향으로 이동하고, MSE가 줄지
않으면 스텝을 10분의 1로 줄여 다시 계산합니다. `--optimizer adam`으로 Adam도
선택할 수 있습니다.
MSE는 fractional 회로 출력과 목표 비트 사이의 손실입니다. 이것이 낮아져도
이진화한 메시지의 해시가 목표와 일치한다는 뜻은 아닙니다.

국소 Hessian은 각 출력에 대한 입력 gradient를 역전파해 Jacobian `J`를 만들고,
`2/256 · JᵀJ`로 계산합니다. 각 ReLU 활성화 구간에서 출력은 affine이므로 이
식이 해당 구간의 MSE Hessian입니다. ReLU 입력이 정확히 0인 곳에서는 미분값
0을 택한 국소 Hessian을 사용합니다. 출력 256개마다 역전파를 수행하므로 전체
`512×512` 행렬을 만드는 데 시간이 걸립니다.

```bash
uv run python sha256_nn_mpmath_hessian.py \
  --precision 113 --input-json fractional_mpmath_train_113_linesearch.json \
  --output fractional_mpmath_hessian_113.json
```

한 가중치의 오차에 따라 입력 gradient가 변하는 양은 다음으로 측정합니다.
이 실험은 `abc`의 올바른 패딩 블록과 digest를 고정하고,
`xor_and.weight_in[0][0]`의 값 1에 `10⁻¹`부터 `10⁻³⁴`까지 더합니다.
기준점의 MSE와 입력 gradient가 모두 0인지 검사한 뒤 각 perturbation의 MSE,
gradient 최댓값과 L2 norm을 저장합니다.

```bash
uv run python sha256_nn_mpmath_weight_sweep.py \
  --precision 113 --output weight_perturbation_113.json
```

IEEE binary256과 같은 237비트 유효숫자로도 실행할 수 있습니다. mpmath는
지수 범위와 특수값 처리를 IEEE binary256과 똑같이 모사하지는 않습니다.
아래 명령은 `10⁻¹`부터 `10⁻⁶⁸`까지 가중치 실험을 실행합니다.
결과는 선택한 정밀도를 보존할 수 있는 십진 자릿수로 저장됩니다.

```bash
uv run python sha256_nn_mpmath_weight_sweep.py --float256 \
  --end-exponent 68 --output weight_perturbation_237_1_68.json

# 순전파·역전파, 입력 최적화, Hessian에도 같은 정밀도를 사용할 수 있음
uv run python sha256_nn_mpmath.py --precision 237 --output fractional_mpmath_237.json
uv run python train_sha256_nn_mpmath.py --precision 237 --output training_237.json
uv run python sha256_nn_mpmath_hessian.py --precision 237 --output hessian_237.json
```

유효숫자 자체를 256비트로 원하면 `--precision 256`을 사용합니다.
입력 가중치 1 부근의 237비트 간격은 약 `9.06×10⁻⁷²`입니다. 요청한 변화량과
실제로 표현된 변화량이 다를 수 있어 결과 파일에 두 값을 모두 기록합니다.

`--mode message`는 64바이트 원본 메시지에 패딩 블록을 추가해 두 블록을
계산합니다. `--mode compression`은 제공된 512개 값을 한 블록으로 계산하며,
이 모드의 `--message-hex` 편의 옵션만 0..55바이트 메시지를 먼저 패딩합니다.
`--input-json`은 512개 값의 배열 또는 `input` 필드를 가진 객체를 받습니다.

```python
import hashlib
from sha256_nn_mpmath import FractionalSHA256MP, bytes_to_bits

model = FractionalSHA256MP(precision=128)
x = [model.ctx.mpf("0.5")] * 512
target = bytes_to_bits(hashlib.sha256(b"abc").digest())
result = model.evaluate(x, target)
print(result.loss)       # mean((output - target)**2)
print(result.gradient)   # MSE에 대한 512개 입력의 gradient

# 순전파만: model.forward(x)
# 임의 출력 gradient의 역전파: model.backward(x, grad_output)
# backward는 순전파를 다시 계산해 J(x).T @ grad_output을 반환합니다.
```

입력은 십진 문자열이나 해당 모델의 `ctx.mpf` 값으로 전달하면 float64를
거치지 않습니다. Python float로 전달하면 이미 손실된 정밀도는 복구되지 않습니다.
JSON의 숫자는 십진 표기 그대로 읽고, 결과의 수치 배열은 충분한 자리수의
문자열로 저장합니다. 과거 float 로그의 십진 표기를 고정밀로 읽은 값은 원래
이진 부동소수점 값과 미세하게 다를 수 있습니다.

높은 정밀도는 gradient 폭주 자체를 없애지 않습니다. 실수 입력에서의 결과와
gradient는 이 ReLU 확장 회로에 대한 값이며, 이진화 후의 해시 일치를 보장하지
않습니다. 비교 테스트는 게이트의 ReLU 경계, 덧셈의 수치 미분, 두 모드의
`hashlib` 대조, 소수 입력의 전체 PyTorch 순전파·역전파 대조를 포함합니다.
