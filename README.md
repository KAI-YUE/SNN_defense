### Stochastic Neural Network Defense

#### Prerequisites

- install Python packages
    ```bash
    pip3 install -r requirements.txt
    ```

- Download the pretrained models and put them under `model_zoos` ([link](https://huggingface.co/erickyue/rog_modelzoo/tree/main))


- The images for a minimal runnable example has been included under `data` folder. The ImageNet validation dataset can be used for a full test. 

<br />

#### Example

- Use SNN defense under FedAvg aggregation:
  ```bash
  python3 main.py
  ```

- Do not use SNN defense:
  ```bash
  python3 no_defense.py
  ```


#### Note
- You can change the settings in the configuration file. For example, use a different $\tau$

```bash
tau: 20
```