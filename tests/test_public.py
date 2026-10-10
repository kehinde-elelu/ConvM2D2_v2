import math
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from conformalmos.calibration import quantile, intervals
from conformalmos.model import PredictionHead
from conformalmos.predict import preprocess

@pytest.mark.parametrize("n,alpha,rank", [(225,.05,215),(165,.05,158),(25,.05,25),(9,.1,9)])
def test_corrected_rank(n, alpha, rank):
    radius, actual = quantile(np.arange(n, dtype=float), alpha)
    assert actual == rank
    assert radius == rank-1

def test_insufficient_calibration_returns_full_mos_range():
    radius, rank = quantile([.1,.2], .05)
    assert math.isinf(radius) and rank == 3
    lo, hi = intervals([1.2,4.8],radius)
    assert np.array_equal(lo,[1,1]) and np.array_equal(hi,[5,5])

def test_historical_mode_is_explicit():
    q, rank = quantile([0,1,2,3],.1,"historical")
    assert q == pytest.approx(2.7) and rank is None

@pytest.mark.parametrize("values", [[],[-1],[float("nan")]])
def test_invalid_residuals(values):
    with pytest.raises(ValueError):
        quantile(values)

def test_real_checkpoint_loads():
    c = torch.load(Path(__file__).resolve().parents[1]/"assets/conformalmos_m2d_head.pt", weights_only=True, map_location="cpu")
    head = PredictionHead().eval()
    head.load_state_dict(c["head_state"], strict=True)
    with torch.no_grad():
        result = head(torch.zeros(2,768))
    assert torch.isfinite(result).all() and ((result>=1)&(result<=5)).all()

def test_stereo_resampling_and_padding(tmp_path):
    wav = tmp_path/"stereo.wav"
    sf.write(wav,np.column_stack([np.ones(8000)*.1,np.ones(8000)*.3]),8000,subtype="FLOAT")
    x = preprocess(wav)
    assert x.shape == (80000,)
    assert x[200:15800].mean().item() == pytest.approx(.2,abs=1e-4)
    assert torch.count_nonzero(x[16000:]) == 0

def test_truncation(tmp_path):
    wav = tmp_path/"long.wav"
    sf.write(wav,np.zeros(96000),16000)
    assert len(preprocess(wav)) == 80000

@pytest.mark.parametrize("overlap", [False,True])
def test_calibration_cli_disjointness(tmp_path, overlap):
    import csv, json, subprocess, sys
    root = Path(__file__).resolve().parents[1]
    source = tmp_path/'calibration.csv'
    with source.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['path','predicted_mos','mos'])
        for i in range(25):
            w.writerow([str(tmp_path/f'c{i}.wav'),3.,3.+i/100])
    evaluation = tmp_path/'evaluation.txt'
    evaluation.write_text(str(tmp_path/('c0.wav' if overlap else 'heldout.wav'))+'\n')
    output = tmp_path/'radius.json'
    result=subprocess.run([sys.executable,'-m','conformalmos.calibrate','--csv',str(source),
        '--checkpoint',str(root/'assets/conformalmos_m2d_head.pt'),
        '--evaluation-paths',str(evaluation),'--alpha','.05','--output',str(output)],capture_output=True,text=True,cwd=root)
    if overlap:
        assert result.returncode != 0 and not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        data=json.loads(output.read_text())
        assert data['rank']==25 and data['q_hat']==pytest.approx(.24)
        assert data['n_calibration']==25

def test_interval_clipping():
    lower, upper=intervals([1.1,4.9],.8)
    assert np.allclose(lower,[1,4.1])
    assert np.allclose(upper,[1.9,5])
