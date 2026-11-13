"""Stage 2: train on 128 PKLot train crops and validate on 64 PKLot valid crops.
No test/online crops, and no production deployment.
"""
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

SOURCE = Path('/kaggle/input/datasets/nguynngthibo/smart-parking-stage2-smoke')
WORKDIR = Path('/kaggle/working/stage2_crops')
OUTPUT = Path('/kaggle/working')


class ParkingCrops(Dataset):
    def __init__(self, root, records):
        self.root = Path(root)
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record['file']
        with Image.open(path) as img:
            image = self.transform(img.convert('RGB'))
        label = int(record['label'])
        if label not in (0, 1):
            raise ValueError(f'Invalid class label: {label}')
        return image, label


def prepare_data():
    print('STAGE2_BEGIN', flush=True)
    print('INPUT_PATH:', SOURCE, flush=True)
    if not SOURCE.is_dir():
        raise RuntimeError(f'Kaggle dataset not attached: {SOURCE}')

    manifest_path = SOURCE / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'Missing manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['stage'] == 'stage2-smoke'
    assert manifest['online_crops_included'] == 0
    assert manifest['test_crops_included'] == 0
    assert len(manifest['train']) == 128
    assert len(manifest['valid']) == 64
    print('STAGE2_MANIFEST_PASS', flush=True)

    records = manifest['train'] + manifest['valid']
    names = [r['file'] for r in records]
    expected = set(names)
    assert len(expected) == len(names) == 192, 'Duplicate file paths'
    for name in expected:
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.parts[0] in ('train', 'valid')

    # Kaggle automatically extracts ZIP files uploaded as Dataset assets.
    # The files might be mounted directly or in a directory named 'crops'.
    candidates = [SOURCE, SOURCE / 'crops', SOURCE / 'crops.zip']
    data_root = next(
        (base for base in candidates
         if base.is_dir() and all((base / name).is_file() for name in expected)),
        None,
    )

    if data_root is None:
        archive = SOURCE / 'crops.zip'
        if archive.is_file() and zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                members = set(z.namelist())
                assert members == expected, 'Archive files differ from manifest'
                assert z.testzip() is None, 'Corrupt ZIP'
                WORKDIR.mkdir(parents=True, exist_ok=True)
                z.extractall(WORKDIR)
            data_root = WORKDIR

    if data_root is None:
        # Show safe relative paths for debugging, not secrets.
        visible = sorted(str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file())
        print('DATASET_VISIBLE_FILES:', visible[:30], flush=True)
        raise RuntimeError('Missing crop images after Kaggle dataset processing')

    print('STAGE2_DATA_ROOT:', data_root, flush=True)
    print('STAGE2_CROPS_FOUND_PASS:', len(expected), flush=True)
    return manifest, data_root


def train_one_epoch(manifest, data_root):
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.manual_seed(42)
    device = torch.device('cuda')
    gpu = torch.cuda.get_device_name(0)
    print('CUDA_AVAILABLE: True', flush=True)
    print('GPU:', gpu, flush=True)

    train_loader = DataLoader(ParkingCrops(data_root, manifest['train']),
                              batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(ParkingCrops(data_root, manifest['valid']),
                              batch_size=16, shuffle=False, num_workers=0)
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    losses = []
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    average_loss = sum(losses) / len(losses)
    print('STAGE2_TRAIN_EPOCH_PASS', average_loss, flush=True)

    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in valid_loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predicted == labels).sum())
            total += len(labels)
    accuracy = correct / total
    print('STAGE2_VALIDATION_PASS', accuracy, flush=True)
    return {'gpu': gpu, 'train_loss': average_loss, 'valid_accuracy': accuracy}


def main():
    manifest, data_root = prepare_data()
    metrics = train_one_epoch(manifest, data_root)
    report = {
        'stage': 'stage2-smoke',
        'model': 'mobilenet_v3_small',
        'weights': 'random_initialization',
        'epochs': 1,
        'production_approved': False,
        'train_samples': 128,
        'valid_samples': 64,
        'online_crops_included': 0,
        'test_crops_included': 0,
        'git_commit': manifest['git_commit'],
        'github_run_id': manifest['github_run_id'],
        'data_dvc_sha256': manifest['data_dvc_sha256'],
        **metrics,
    }
    path = OUTPUT / 'stage2_report.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('STAGE2_GPU_TRAIN_PASS', flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
