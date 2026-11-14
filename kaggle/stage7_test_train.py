"""TEST ONLY: Kaggle training on PKLot train; no online data or Production promotion."""
import hashlib
import json
import math
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, resnet18
from PIL import Image

DATASET_NAME = '__STAGE7_DATASET_NAME__'
RUN_ID = '__EXPECTED_RUN_ID__'
SOURCE = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
UNPACK = OUTPUT / 'stage7_unpack'


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Crops(Dataset):
    def __init__(self, root, records):
        self.root, self.records = root, records
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        with Image.open(self.root / row['file']) as im:
            return self.transform(im.convert('RGB')), int(row['label'])


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self, x):
        return self.conv(x)


class OptimizedPKLotCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1, self.block2 = ConvBlock(3, 32), ConvBlock(32, 64)
        self.block3, self.block4 = ConvBlock(64, 128), ConvBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True), nn.Dropout(.4), nn.Linear(128, 2))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(self.global_pool(x))


def build_model(name):
    if name == 'mobilenetv3':
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
        return model
    if name == 'resnet18':
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    if name == 'custom_cnn':
        return OptimizedPKLotCNN()
    raise ValueError(name)


def load_payload():
    paths = sorted(SOURCE.rglob('manifest.json'))
    for path in paths:
        if DATASET_NAME not in str(path.parent):
            continue
        m = json.loads(path.read_text(encoding='utf-8'))
        if m.get('stage') != 'stage7-test-only' or str(m.get('github_run_id')) != RUN_ID:
            continue
        if m['dataset_name'] != DATASET_NAME or m['origin'] != 'pklot_train_test_only' or m.get('test_only') is not True:
            raise ValueError('Stale or unapproved input')
        names = [r['file'] for r in m['train'] + m['valid']] + [v['file'] for v in m['checkpoints'].values()]
        if len(names) != len(set(names)) or set(m['checkpoints']) != {'mobilenetv3', 'resnet18', 'custom_cnn'}:
            raise ValueError('Duplicate paths / model mismatch')
        for name in names:
            p = Path(name)
            if p.is_absolute() or '..' in p.parts or p.parts[0] not in ('train', 'valid', 'checkpoint'):
                raise ValueError('Unsafe path')
        root = None
        for candidate in (path.parent, path.parent / 'payload'):
            if all((candidate / n).is_file() for n in names):
                root = candidate
                break
        if root is None:
            archive = path.parent / 'payload.zip'
            if not zipfile.is_zipfile(archive):
                raise ValueError('Payload missing')
            with zipfile.ZipFile(archive) as z:
                if set(z.namelist()) != set(names) or z.testzip() is not None:
                    raise ValueError('Payload archive mismatch')
                UNPACK.mkdir(parents=True, exist_ok=True)
                z.extractall(UNPACK)
            root = UNPACK
        for model, info in m['checkpoints'].items():
            if digest(root / info['file']) != info['sha256']:
                raise ValueError(f'{model} source checkpoint integrity failure')
        for r in m['train'] + m['valid']:
            if digest(root / r['file']) != r['sha256']:
                raise ValueError('Image integrity failure')
        if len(m['valid']) < 200 or len(m['train']) < 32:
            raise ValueError('Insufficient new training or validation data')
        print('STAGE7_CURRENT_RUN_DATASET_PASS', flush=True)
        print('STAGE7_CHECKSUMS_PASS', flush=True)
        return root, m
    raise RuntimeError('Dataset for this exact GitHub run not mounted')


def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return correct / total


def train_one(name, root, manifest, device):
    torch.manual_seed(42)
    model = build_model(name)
    cp = manifest['checkpoints'][name]
    state = torch.load(root / cp['file'], map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise TypeError('Expected state_dict')
    state = state.get('state_dict', state.get('model_state_dict', state))
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    train_loader = DataLoader(Crops(root, manifest['train']), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(Crops(root, manifest['valid']), batch_size=32, shuffle=False, num_workers=0)
    baseline = accuracy(model, val_loader, device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    candidate_acc = accuracy(model, val_loader, device)
    if not math.isfinite(candidate_acc):
        raise ValueError('Bad candidate accuracy')
    path = OUTPUT / f'stage7_candidate_{name}.pth'
    torch.save(model.cpu().state_dict(), path)
    del model
    torch.cuda.empty_cache()
    print('STAGE7_' + name.upper() + '_TRAIN_PASS', flush=True)
    return {'baseline_valid_accuracy': baseline, 'candidate_valid_accuracy': candidate_acc,
            'train_loss': sum(losses) / len(losses), 'source_checkpoint_sha256': cp['sha256'],
            'candidate_checkpoint_sha256': digest(path)}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Stage 7 requires Kaggle GPU')
    root, m = load_payload()
    device = torch.device('cuda')
    models = {name: train_one(name, root, m, device) for name in ('mobilenetv3', 'resnet18', 'custom_cnn')}
    result = {'stage': 'stage7-test-only', 'test_only': True, 'github_run_id': m['github_run_id'],
              'git_commit': m['git_commit'], 'approved_dvc_sha256': m['approved_dvc_sha256'],
              'data_dvc_sha256': m['data_dvc_sha256'], 'train_samples': len(m['train']),
              'valid_samples': len(m['valid']), 'gpu': torch.cuda.get_device_name(0),
              'production_approved': False, 'models': models}
    (OUTPUT / 'stage7_train_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('STAGE7_TEST_ONLY_GPU_THREE_MODELS_PASS', flush=True)


if __name__ == '__main__':
    main()
