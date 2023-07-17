# **GNetDet** Inference Program on Chip SIT501.



**Requirments：**

```shell
pip install opencv-python pyyaml
```

   

**Compile [Optional]：**

```shell
python3 -m compileall ./
```

​    

**Examples:**

- **Run detection on image:**

  - ```shell
    python3 GNetDet.py image --model=vehicle.model --input=test/imgs/test.jpg
    
    # If input is a directory, then detect all images under this directory
    python3 GNetDet.py image --model=vehicle.model --input=test/imgs
    
    # If output path is given, then all results will be saved
    python3 GNetDet.py image --model=vehicle.model --input=test/imgs --output=test/results
    ```

- **Run detection on video:**

  - ```shell
    python3 GNetDet.py video --model=vehicle.model --input=test.mp4
    ```

- **Run real time object detection on camara:**

  - ```shell
    python3 GNetDet.py camara --model=vehicle.model
    ```



**Some arguments:**

- **type**
  - Detection type, must be **image**, **video** or **camara**.

- **-m/--model**      
  - Path of model，*default* is  `./out.model`.
- **-i/--input** 
  - Path of input source. REQUIRED  if detection type is  **image** or **video**.
     - Could be a specific file path, eg. `./test.jpg`;
     - Or a directory. In such case all *images* or *videos* under the directory will be detected.
- **-o/--output**
     - Path to output.  Currently only usefull if *type* is *image*.
     - Could be a specific file path, eg. `./out-test.jpg`. Avoid duplicated name as input.
     - Or a directory. The results will be save in the directory with same name as input suffixed by `out-`, eg. `./test-output/out-test.jpg`.
- **-f/--fancy**
  - Fancier bounding boxes.



**More details：**

- Refer to docs:

  - ```
    python3 GNetDet.py --help
    ```

- Take a look at `GNetDet.yaml`.
- Inspec codes in `config.py`.
