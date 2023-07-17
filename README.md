# GNetDet Pytorch Training Toolkit



**Data preparation:**

See **DSCollection** documentation for dataset  generation, extraction or other data manipulations, which could save your time when you organize any dataset into Pascal VOC dataset format.



**Step 0: Generate dataset meta data**

Edit `config.py` properly, change parameters for your dataset:

- **VOC_CLASSES**: A list of strings, class names from dataset
- **DS_NAME**: Dataset directory name under `data` directory



Then run the following script to generate meta data:  

```python
python tools/xml_2_txt.py
```



**Step 1: Training a FP32 model**

A decent FP32 model is training on this step. You can modify any hyperparameters on training-phase as described in `config.py`. There are many of them which you may frequently changed across different difference datasets:

- learning_rate
- batch_size
- num_epochs
- warmup_epochs
- step_size
- gamma
- warmup_epochs
- ... and more

Refer to `config.py` and `gnetmdk/config/config.py` for more details.

Use the following script to start phase 1 training, you can specify any hyperparameters using CLI arguments as well to override those with same name in `config.py` (would be helpfull when you are training on a remote server).

```python
python train-dist.py --step 1 [specify any hyperparameters to override config.py]

# example
python train-dist.py --step 1 \
learning-rate 1e-3 \
checkpoint-path checkpoint/step0/best.pth \
num-epochs 600 \
step-size 400 \
gamma 0.1 \
warmup-epochs 100 
```



**Step 2: Quantize Conv Layers**

Next you will need to quantize all conv layers. Refer GNetDet specifications and documentation for more details about quantization in this step and how to avoid some common pitfalls started from this step.

```python
python train-dist.py --step 2 [hyperparameters]

# example
python train-dist.py --step 2 \
learning-rate 1e-4 \
checkpoint-path checkpoint/step1/best.pth \
num-epochs 800 \
step-size 400 \
gamma 0.1 \
warmup-epochs 200
```



**Step 3: Calibration activations**

After step 2 training, you got a model with quantized conv layers, which may not perform as good as the phase 1 FP32 model does, but hopes does not differ to much. Before you quantize activations, you need to calibrate them, run the following script.

```python
python train-dist.py --step 2 --cal checkpoint-path checkpoint/step2/best.pth
```



**Step 4: Quantize Activations**

Another separate step to quantize activation values, as well as conv layers:

```python
python train-dist.py --step 3 [hyperparameters]

# example
python train-dist.py --step 3 \
learning-rate 1e-5 \
checkpoint-path checkpoint/step2/best.pth \
num-epochs 800 \
step-size 400 \
gamma 0.1 \
warmup-epochs 200
```

Also, the performance degradation may not be too much, 1~10% degradation is always acceptable.



**Step 5: Quantize to Chip**

 The final training step is turn on quant-31, which is required by 5801 chip. 

```python
python train-dist.py --step 4 [hyperparameters]

# example
python train-dist.py --step 4 \
learning-rate 1e-5 \
checkpoint-path checkpoint/step3/best.pth \
num-epochs 500 \
step-size 200 \
gamma 0.1 \
warmup-epochs 100
```

If the training successfull, you may quickly see the model perform as close as step 3. During this step, the evaluation may drastically drop down and quickly retrieve, this phenomenon can be clearly seen when you turn on Tensorboard during training, but don't worry about that, you can manually stop the training procedure as you want if you see the mAP is good enough.



**Convert to Chip Model**

Run the script and grab a coffee.

```python
python tools/convert_to_chip.py
```



**Run inference using pyGNetDet**

Once the chip model converted (usually saved `./model/out.model` subdirectory), copy to `./pyGNetDet`, then edit `GNetDet.yaml` file, change CAP value to the last relu cap value in `./model/relu_cap.txt`, grab some test images or videos and run inference:

```python
python GNetDet.py image -m out.model -i test/imgs -o test/output
```

This script will load the model onto the chip and save the inference images into output directory.
