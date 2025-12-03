# ShapeNet-Audio Benchmark

**Zero-shot 3D shape prediction from impact sounds**

## Dataset
- 50k 3D objects from ShapeNet
- Synthesized impact sounds (impact audio)
- Pre-computed features via PointNet

## Task
Given an audio clip, predict the 3D shape category.

## Expected Result
- Target: 58.9% Recall@1
- Current: TBD (run `python run_benchmark.py`)

## How to Run
```bash
cd benchmarks/
python shapenet_audio.py

@techreport{shapenet-audio-2025,
  title={ShapeNet-Audio: Zero-shot Shape from Sound},
  author={AntonL1169},
  year={2025},
  url={https://github.com/AntonL1169/alms-protocol}
}
