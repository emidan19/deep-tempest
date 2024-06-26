# End-to-End Method

<img src="end-to-end.png"/>

## Usage Guide

In general, the options to use (reference/degraded image folders, network models, output directory, etc.) are located in[end-to-end/options](../end-to-end/options).

### Inference and Evaluation

To run inference, you need to edit the file [end-to-end/options/train_drunet.json](../end-to-end/options/train_drunet.json) and, once the changes are made, execute:

```shell
python main_test_drunet.py
```
This command will output a new directory with the inferences from the input directory.

To evaluate a directory with images (both reference and model's inference), you need to edit the file [end-to-end/options/evaluation.json](../end-to-end/options/evaluation.json) and, once the changes are made, execute:
```shell
python tempest_evaluation.py
```

### Training

**Note: Before executing the following command, you must select which type of data to use for training**

#### Training with Real Data

To train with real data, the file [end-to-end/options/train_drunet.json](../end-to-end/options/train_drunet.json) must have the value __"drunet_finetune"__ in the *dataset_type* field (datasets-->train).

#### Training with Synthetic Data

To train with synthetic data, the file end-to-end/options/train_drunet.json](../end-to-end/options/train_drunet.json) must have the value __"drunet"__ in the *dataset_type* field (datasets-->train).

Once the data type was selected, use the following command to train the network:

```shell
python main_train_drunet.py
```
### Generating Synthetic Captures

For synthetic captured images generation, first configure the options on [tempest_simulation.json](end-to-end/options/tempest_simulation.json) file. Be sure to include the path to the folder containing the images to run the simulation of direct capturing image from the EME of a monitor. Then run the following command:

```shell
python folder_simulation.py
```
Which outputs the synthetic captured in the specified folder.