# Usage

## Prerequirements

- docker
- docker compose
- nvidia-container-toolkit

## Step 1. 
```bash
docker compose up -d
docker compose ps
docker compose exec -it tetris /bin/bash
```

## Step 2. On TCP server
```python
python ./server.py
```

## Step 3. Check TCP server connection 
```python
python test_tetris_env.py
```

## Step 4. Tensorboard (Optional)
```bash
tensorboard --logdir <your log dir path>
```

e.g.

```bash
tensorboard --logdir logs/sb3_log
```

## Step 5. Train your model
```python
python app.py
```
