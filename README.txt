ENV: Python 3.6, PyTorch 0.4.0

RUNNING EXAMPLES:
python3 main.py --epochs 10
python3 main.py --epochs 10 --dropout 0.5
python3 main.py --epochs 10 --weight-decay 1e-3
python3 main.py --epochs 10 --batch-norm True
python3 main.py --saved-weights weights.pt