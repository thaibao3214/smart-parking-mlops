"""A one-epoch Kaggle GPU train smoke-test. NOT retraining from v1 or production-approved."""
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

SOURCE = Path('/kaggle/input/datasets/nguynngthibo/smart-parking-stage2-smoke')
OUT = Path('/kaggle/working')


class Crops(Dataset):
    def __init__(self, root, records):
        self.root = root
        self.records = records
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        path = self.root / rec['file']
        with Image.open(path) as im:
            image = self.transform(im.convert('RGB'))
        return image, int(rec['label'])


def main():
    torch.manual_seed(42)
    print('STAGE2_BEGIN', flush=True)
    assert SOURCE.is_dir(), f'Not found: {SOURCE}'
    meta = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    assert meta['online_crops_included'] == 0
    assert meta['test_crops_included'] == 0
    assert len(meta['train']) == 128 and len(meta['valid']) == 64
    assert torch.cuda.is_available(), 'GPU unavailable'
    device = torch.device('cuda')
    train_loader = DataLoader(Crops(SOURCE, meta['train']), batch_size=16, shuffle=True, num_workers=0)
    valid_loader = DataLoader(Crops(SOURCE, meta['valid']), batch_size=16, num_workers=0)
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    model.eval()
    correct = 0
    with torch.inference_mode():
        for images, labels in valid_loader:
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += int((predictions == labels).sum())
    report = {
        'stage': 'stage2-smoke',
        'model': 'mobilenet_v3_small',
        'weights': 'random_initialization',
        'production_approved': False,
        'train_samples': len(meta['train']),
        'valid_samples': len(meta['valid']),
        'train_loss_sum': total_loss,
        'valid_accuracy': correct / len(meta['valid']),
        'git_commit': meta['git_commit'],
        'github_run_id': meta['github_run_id'],
        'data_dvc_sha256': meta['data_dvc_sha256'],
        'gpu': torch.cuda.get_device_name(0),
    }
    (OUT / 'stage2_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('STAGE2_GPU_TRAIN_PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
