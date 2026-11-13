"""Stage 3: one-epoch MobileNetV3 fine-tuning of an original v1 .pth state_dict.
This is a GPU/integrity validation only: no test or online data; no production promotion.
"""
import hashlib
import json
import math
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

EXPECTED_DATASET_NAME = '__STAGE3_DATASET_NAME__'
EXPECTED_RUN_ID = '__EXPECTED_RUN_ID__'
INPUT_ROOT = Path('/kaggle/input')
WORKDIR = Path('/kaggle/working/stage3_unpack')
OUTPUT = Path('/kaggle/working')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ParkingCrops(Dataset):
    def __init__(self, root, records):
        self.root, self.records = Path(root), records
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        entry = self.records[idx]
        with Image.open(self.root / entry['file']) as img:
            image = self.transform(img.convert('RGB'))
        label = int(entry['label'])
        if label not in (0, 1):
            raise ValueError(f'Invalid class label: {label}')
        return image, label


def locate():
    if not INPUT_ROOT.is_dir():
        raise RuntimeError('Kaggle input directory missing')
    paths = sorted(INPUT_ROOT.rglob('manifest.json'))
    print('AVAILABLE_MANIFESTS:', [str(p) for p in paths[:20]], flush=True)
    for path in paths:
        if EXPECTED_DATASET_NAME not in str(path.parent):
            continue
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if (manifest.get('stage') == 'stage3-finetune'
                and str(manifest.get('github_run_id')) == EXPECTED_RUN_ID):
            print('STAGE3_CURRENT_RUN_DATASET_PASS:', path.parent, flush=True)
            return path.parent, manifest
    raise RuntimeError('Dataset for current GitHub run not mounted; refusing stale data')


def prepare():
    source, manifest = locate()
    assert manifest['online_crops_included'] == 0
    assert manifest['test_crops_included'] == 0
    assert len(manifest['train']) == 128
    assert len(manifest['valid']) == 64
    crop_names = [r['file'] for r in manifest['train'] + manifest['valid']]
    checkpoint_name = 'checkpoint/mobilenet_v3_small_best.pth'
    expected = set(crop_names + [checkpoint_name])
    assert len(expected) == 193
    for name in expected:
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.parts[0] in ('train', 'valid', 'checkpoint')

    # Kaggle may auto-extract ZIP and may nest the extracted contents.
    candidates = [source, source / 'payload', source / 'crops']
    root = next((candidate for candidate in candidates
                 if candidate.is_dir() and all((candidate / name).is_file() for name in expected)), None)
    if root is None:
        archive = source / 'payload.zip'
        if archive.is_file() and zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                assert set(z.namelist()) == expected, 'Dataset payload differs from manifest'
                assert z.testzip() is None, 'Corrupted archive'
                WORKDIR.mkdir(parents=True, exist_ok=True)
                z.extractall(WORKDIR)
            root = WORKDIR
    if root is None:
        print('VISIBLE_FILES:', [str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()][:30])
        raise RuntimeError('Missing staged images or original checkpoint')
    checkpoint = root / checkpoint_name
    assert sha256(checkpoint) == manifest['v1_checkpoint_sha256'], 'Checkpoint SHA256 mismatch'
    print('STAGE3_CHECKPOINT_SHA256_PASS', flush=True)
    print('STAGE3_CROPS_FOUND_PASS: 192', flush=True)
    return root, checkpoint, manifest


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predicted == labels).sum())
            total += len(labels)
    return correct / total


def main():
    root, checkpoint, manifest = prepare()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; GPU is required')
    torch.manual_seed(42)
    device = torch.device('cuda')
    gpu = torch.cuda.get_device_name(0)
    print('GPU:', gpu, flush=True)
    train_loader = DataLoader(ParkingCrops(root, manifest['train']), batch_size=16,
                              shuffle=True, num_workers=0)
    valid_loader = DataLoader(ParkingCrops(root, manifest['valid']), batch_size=16,
                              shuffle=False, num_workers=0)
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise TypeError('Expected a PyTorch state_dict checkpoint')
    if 'state_dict' in state:
        state = state['state_dict']
    elif 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print('STAGE3_V1_CHECKPOINT_LOAD_PASS', flush=True)
    baseline_accuracy = evaluate(model, valid_loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    candidate_accuracy = evaluate(model, valid_loader, device)
    assert math.isfinite(candidate_accuracy)
    output_checkpoint = OUTPUT / 'stage3_candidate_mobilenetv3.pth'
    torch.save(model.cpu().state_dict(), output_checkpoint)
    report = {
        'stage': 'stage3-finetune',
        'model': 'mobilenet_v3_small',
        'weights': 'v1_checkpoint',
        'epochs': 1,
        'training_approved_for_production': False,
        'production_approved': False,
        'train_samples': 128,
        'valid_samples': 64,
        'online_crops_included': 0,
        'test_crops_included': 0,
        'git_commit': manifest['git_commit'],
        'github_run_id': manifest['github_run_id'],
        'data_dvc_sha256': manifest['data_dvc_sha256'],
        'v1_checkpoint_sha256': manifest['v1_checkpoint_sha256'],
        'candidate_checkpoint_sha256': sha256(output_checkpoint),
        'gpu': gpu,
        'baseline_valid_accuracy': baseline_accuracy,
        'candidate_valid_accuracy': candidate_accuracy,
        'train_loss': sum(losses) / len(losses),
    }
    (OUTPUT / 'stage3_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('STAGE3_FINETUNE_PASS', flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
