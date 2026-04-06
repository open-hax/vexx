# @open-hax/vexx

`vexx` is the local semantic fast lane.

Initial scope:
- `POST /v1/cosine/matrix`
- `POST /v1/cosine/topk`
- explicit CPU / GPU / NPU device selection
- no silent accelerator fallback

Current backend split:
- `GPU` -> ONNX Runtime CUDA path
- `NPU` -> native OpenVINO-backed path through `libvexx_cosine.so`
- `CPU` -> local fallback

## Run

```bash
./scripts/build-native.sh
clojure -M:run
```

## Env

- `VEXX_HOST` default `127.0.0.1`
- `VEXX_PORT` default `8788`
- `VEXX_API_KEY` optional bearer token
- `VEXX_DEVICE` default `AUTO`
- `VEXX_AUTO_ORDER` default `GPU,NPU,CPU`
- `VEXX_REQUIRE_ACCEL` default `false`
- `VEXX_CUDA_DEVICE_ID` default `0`
- `VEXX_NATIVE_LIB_PATH` optional explicit path
- `VEXX_COSINE_MATRIX_MODEL_PATH` optional explicit path

If `VEXX_COSINE_MATRIX_MODEL_PATH` is unset, `vexx` uses its own extracted
`models/cosine_matrix_dynamic.onnx`.

## Native payload

`vexx` vendors the minimum native payload it needs:
- cosine ONNX models extracted from `fork_tales`
- ONNX Runtime headers extracted from `fork_tales`
- ONNX Runtime OpenVINO shared libraries extracted from the official
  `onnxruntime-openvino` wheel

That keeps runtime execution local to `vexx` instead of depending on the
experimental `fork_tales` repo at runtime.
