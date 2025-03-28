# K-Bot V2 

This directory contains task definitions for the K-Bot V2.

By default, actor and critic models are MLPs.

## Naming

The naming convention for the tasks is as follows:

```
<task_name>_<feature>
```

For example, `standing_fixed` is a task that trains the K-Bot V2 using a modified MJCF with fixed arms while `standing_lstm` is a task that trains the K-Bot V2 using a modified LSTM actor.



## Running

Make sure that `ksim-kbot` is installed (by running `pip install -e .` in the root directory).

```bash
python -m ksim_kbot.standing.standing_fixed
``` 

To test the model, run:

```bash
python -m ksim_kbot.deploy.sim --model_path /path/to/model
```

And to deploy the model on the real robot, run:

```bash
python -m ksim_kbot.deploy.real --model_path /path/to/model
```






