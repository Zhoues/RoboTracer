
<h1 align="center">RoboTracer: Mastering Spatial Trace with Reasoning in Vision-Language Models for Robotics</h1>

<h3 align="center">From what you say to where it moves — with RoboTracer</h3>


<p align="center">
  <a href="https://arxiv.org/abs/2512.13660"><img src="https://img.shields.io/badge/arXiv-2512.13660-b31b1b.svg" alt="arXiv"></a>
  &nbsp;
  <a href="https://zhoues.github.io/RoboTracer/"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="Project Homepage"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/JingkunAn/TraceSpatial-Bench"><img src="https://img.shields.io/badge/🤗%20Benchmark-TraceSpatial--Bench-green.svg" alt="Benchmark"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Stay%20tuned-yellow" alt="Weights"></a>
</p>


<div style="text-align: center; background-color: white;">
    <img src="assets/motivation.png" width=100% >
</div>


## 🔥 Updates

[2025-12-23] 🔥🔥🔥 TraceSpatial-Bench is released on [HF](https://huggingface.co/datasets/JingkunAn/TraceSpatial-Bench). Let's evaluate your model's spatial tracing ability!


[2025-12-16] 🔥🔥🔥 We release RoboTracer on [arxiv](https://arxiv.org/abs/2512.13660) and launch the [project page](https://zhoues.github.io/RoboTracer/). It retains all [RoboRefer](https://github.com/zhoues/RoboRefer) (previous version) features while also further supporting multi-step, metric-grounded spatial tracing with explicit reasoning. 


## 🤗 Model Zoo &  Dataset & Benchmark


<table>
  <tr>
    <th>Model/Dataset/Benchmark</th>
    <th>Note</th>
  </tr>

  <tr>
    <td><a href="https://huggingface.co/datasets/JingkunAn/TraceSpatial-Bench">TraceSpatial-Bench</a></td>
    <td> The benchmark for spatial tracing with reasoning. </td>
  </tr>

</table>


## 🔍 Evaluation for TraceSpatial-Bench


1. Open the `Evaluation` folder and download the TraceSpatial-Bench from the [model zoo](#-model-zoo---dataset--benchmark).
    ```bash
    cd Evaluation
    git lfs install
    git clone https://huggingface.co/datasets/JingkunAn/TraceSpatial-Bench
    ```

2. Run the API server for RoboTracer (Coming soon) or General VLM (e.g., Gemini, Qwen, etc.).
    - For RoboTracer, you can use the API server provided in the [API](https://github.com/Zhoues/RoboTracer/tree/main/API) folder (Coming soon).
    - For General VLM, you can use the API server provided in the [query_model.py](https://github.com/Zhoues/RoboTracer/blob/main/Evaluation/query_model.py) file. Make sure the api key and base url are set correctly.

3. Run the evaluation script.
   - `task_name`: `2D`, `3D`, or `all` (evaluate all tasks)
   - `model_name` selects which model to use:
     - `RoboTracer_Intrinsics_Depth`: infer with Intrinsics + Depth + RGB using Spaital Encoder.
     - `RoboTracer_Intrinsics`: infer with Intrinsics + RGB using Spaital Encoder.
     - `RoboTracer_RGB`: RGB only inference without spatial Encoder.
     - `RoboTracer`: RGB only inference with spatial Encoder.
     - `Gemini3Pro`: Gemini 3 Pro model
     - ...

    ```bash
    cd Evaluation

    # For RoboTracer
    python test_benchmark.py \
    --model_name RoboTracer_Intrinsics_Depth \
    --task_name all \
    --url http://127.0.0.1:25547

    # For General VLM
    python test_benchmark.py \
    --model_name Gemini3Pro \
    --task_name all
    ```

4. Summarize the results.
    - The `model_name` must be the same as the one used in the evaluation script (we provide the `RoboTracer_Intrinsics_Depth` and `Gemini3Pro` results in the `Evaluation/outputs` folder for reference).
    - The `task_name` can be `2D`, `3D`, or `all` to summarize the results for the corresponding task.

    ```bash
    cd Evaluation

    # For RoboTracer
    python summarize_acc.py \
    --model_name RoboTracer_Intrinsics_Depth \
    --task_name all

    # For General VLM
    python summarize_acc.py \
    --model_name Gemini3Pro \
    --task_name all
    ```

## 🕶️Overview

### The Overview of RoboTracer

We introduce RoboTracer, **the first 3D-aware reasoning VLM** for multi-step metric-grounded spatial tracing with explicit reasoning.

<div align="center"> 
    <img src="assets/pipeline.png" alt="Logo" style="width=100%;vertical-align:middle">
</div>


### The Overview of the TraceSpatial Dataset and its Generation Pipeline

We present TraceSpatial, a dataset can enable general VLMs to adapt to spatial tracing tasks, with **4.5M data samples (~30M QA pairs)** from 2D/3D/Video sources, spanning **outdoor/indoor/tabletop scenes** and containing **complex reasoning processes (up to 9 steps)**.


<div align="center"> 
    <img src="assets/dataset.png" alt="Logo" style="width=100%;vertical-align:middle">
</div>


## TODO
- [x] Release TraceSpatial-Bench evaluation code (About 2 week).
- [ ] Release the SFT-trained 2B RoboTracer model and inference code (About 1 month).
- [ ] Release the SFT-trained 8B RoboTracer model (About 2 months).
- [ ] Release the TraceSpatial Dataset and SFT training code (About 2 months).
- [ ] Release the RFT-trained RoboTracer model and training code (Maybe 2 months or more).
- [ ] Release the Dataset Generation Pipeline (Maybe 2 months or more).


## Contact
If you have any questions about the code or the paper, feel free to email Enshen (`zhouenshen@buaa.edu.cn`) Yibo (`leeibo@buaa.edu.cn`), and Jingkun (`anjingkun02@gmail.com`). 






## Acknowledgment
- This repository is built upon the codebase of [NVILA](https://github.com/NVlabs/VILA), [RoboRefer](https://github.com/zhoues/RoboRefer), [MapAnything](https://github.com/facebookresearch/map-anything), [R1-V](https://github.com/Deep-Agent/R1-V).

- We acknowledge [OpenImage](https://storage.googleapis.com/openimages/web/index.html), [CA-1M](https://github.com/apple/ml-cubifyanything), [ScanNet](http://www.scan-net.org), [DROID](https://droid-dataset.github.io/), [AgiBot-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta), [RoboTwin 2.0](https://github.com/robotwin-Platform/robotwin) for their data and assets.








## 📑 Citation

If you find RoboTracer, TraceSpatial, and TraceSpatial-Bench useful for your research, please cite using this BibTeX:
```
@article{zhou2025robotracer,
    title={RoboTracer: Mastering Spatial Trace with Reasoning in Vision-Language Models for Robotics},
    author={Zhou, Enshen and Chi, Cheng and Li, Yibo and An, Jingkun and Zhang, Jiayuan and Rong, Shanyu and Han, Yi and Ji, Yuheng and Liu, Mengzhen and Wang, Pengwei and others},
    journal={arXiv preprint arXiv:2512.13660},
    year={2025}
}
```
