"""Stage 4 GPU smoke: fine-tune original ResNet18 and Custom CNN v1 state_dicts.
Does not use online/test images or modify MLflow Production.
"""
import hashlib
import json
import math
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

DATASET_NAME = '__STAGE4_DATASET_NAME__'
RUN_ID = '__EXPECTED_RUN_ID__'
INPUT_ROOT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
UNPACK = OUTPUT / 'stage4_unpack'
MODELS = {
    'resnet18': 'checkpoint/resnet18_best.pth',
    'custom_cnn': 'checkpoint/optimized_custom_cnn_best.pth',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Crops(Dataset):
    def __init__(self, root, rows):
        self.root = Path(root)
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row['file']) as im:
            image = self.transform(im.convert('RGB'))
        label = int(row['label'])
        if label not in (0, 1):
            raise ValueError('Invalid label')
        return image, label


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class OptimizedPKLotCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(.4), nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(self.global_pool(x))


def build(name):
    if name == 'resnet18':
        net = resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, 2)
        return net
    if name == 'custom_cnn':
        return OptimizedPKLotCNN()
    raise ValueError(name)


def locate_payload():
    manifests = sorted(INPUT_ROOT.rglob('manifest.json')) if INPUT_ROOT.exists() else []
    print('AVAILABLE_MANIFESTS:', [str(p) for p in manifests[:25]], flush=True)
    for path in manifests:
        if DATASET_NAME not in str(path.parent):
            continue
        try:
            m = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if m.get('stage') == 'stage4-two-models' and str(m.get('github_run_id')) == RUN_ID:
            print('STAGE4_CURRENT_RUN_DATASET_PASS:', path.parent, flush=True)
            return path.parent, m
    raise RuntimeError('This run dataset is not mounted; rejecting stale data')


def read_payload():
    source, m = locate_payload()
    if len(m['train']) != 128 or len(m['valid']) != 64:
        raise ValueError('Unexpected crop count')
    if m['online_crops_included'] != 0 or m['test_crops_included'] != 0:
        raise ValueError('Test/online data not allowed in smoke test')
    names = [r['file'] for r in m['train'] + m['valid']] + list(MODELS.values())
    if len(names) != len(set(names)) or len(names) != 194:
        raise ValueError('Duplicate/missing entries')
    for name in names:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('train', 'valid', 'checkpoint'):
            raise ValueError(f'Unsafe path {name}')
    expected = set(names)
    root = next((p for p in (source, source / 'payload') if p.is_dir() and
                 all((p / n).is_file() for n in expected)), None)
    if root is None:
        archive = source / 'payload.zip'
        if archive.is_file() and zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                if set(z.namelist()) != expected or z.testzip() is not None:
                    raise ValueError('Bad payload archive')
                UNPACK.mkdir(parents=True, exist_ok=True)
                z.extractall(UNPACK)
            root = UNPACK
    if root is None:
        print('VISIBLE_FILES:', [str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()][:35])
        raise RuntimeError('Missing images/checkpoints in mounted data')
    for key, relative in MODELS.items():
        if sha256(root / relative) != m['v1_checkpoint_sha256'][key]:
            raise ValueError(f'{key} checkpoint hash mismatch')
    print('STAGE4_CHECKPOINT_SHA256_PASS', flush=True)
    print('STAGE4_CROPS_FOUND_PASS: 192', flush=True)
    return root, m


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            out = model(x.to(device)).argmax(1).cpu()
            correct += int((out == y).sum())
            total += len(y)
    return correct / total


def finetune_one(name, root, manifest, device):
    torch.manual_seed(42)
    train = DataLoader(Crops(root, manifest['train']), batch_size=16, shuffle=True, num_workers=0)
    valid = DataLoader(Crops(root, manifest['valid']), batch_size=16, shuffle=False, num_workers=0)
    model = build(name)
    state = torch.load(root / MODELS[name], map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise TypeError('Expected state_dict checkpoint')
    state = state.get('state_dict', state.get('model_state_dict', state))
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print(f'STAGE4_{name.upper()}_V1_LOAD_PASS', flush=True)
    before = accuracy(model, valid, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for x, y in train:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss')
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    after = accuracy(model, valid, device)
    if not math.isfinite(after):
        raise ValueError('Invalid accuracy')
    candidate = OUTPUT / f'stage4_candidate_{name}.pth'
    torch.save(model.cpu().state_dict(), candidate)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'STAGE4_{name.upper()}_FINETUNE_PASS', flush=True)
    return {
        'baseline_valid_accuracy': before,
        'candidate_valid_accuracy': after,
        'train_loss': sum(losses) / len(losses),
        'candidate_checkpoint_sha256': sha256(candidate),
        'v1_checkpoint_sha256': manifest['v1_checkpoint_sha256'][name],
    }


def main():
    root, manifest = read_payload()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    device = torch.device('cuda')
    gpu = torch.cuda.get_device_name(0)
    print('GPU:', gpu, flush=True)
    results = {}
    for name in ('resnet18', 'custom_cnn'):
        results[name] = finetune_one(name, root, manifest, device)
    report = {
        'stage': 'stage4-two-models', 'github_run_id': manifest['github_run_id'],
        'git_commit': manifest['git_commit'], 'data_dvc_sha256': manifest['data_dvc_sha256'],
        'train_samples': 128, 'valid_samples': 64, 'epochs': 1,
        'online_crops_included': 0, 'test_crops_included': 0,
        'production_approved': False, 'gpu': gpu, 'models': results,
    }
    (OUTPUT / 'stage4_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('STAGE4_TWO_MODELS_FINETUNE_PASS', flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
