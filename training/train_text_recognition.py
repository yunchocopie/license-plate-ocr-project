"""
텍스트 인식 모델 학습 파이프라인

CRNN (Convolutional Recurrent Neural Network) 기반
한국 번호판 텍스트 인식 모델 학습
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.augmentation import PlateAugmentation, AUGMENTATION_PRESETS
import config


class PlateTextDataset(Dataset):
    """번호판 텍스트 인식 데이터셋"""

    def __init__(self, data_dir, labels_file, charset, augmentation=None, max_len=10):
        """
        데이터셋 초기화

        Args:
            data_dir: 이미지 디렉토리
            labels_file: 레이블 JSON 파일
            charset: 문자 집합
            augmentation: 증강 객체
            max_len: 최대 텍스트 길이
        """
        self.data_dir = Path(data_dir)
        self.charset = charset
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(charset)}  # 0은 blank
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.augmentation = augmentation
        self.max_len = max_len

        # 레이블 로드
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

        self.samples = [(self.data_dir / filename, text)
                       for filename, text in self.labels.items()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text = self.samples[idx]

        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

        # 증강 적용
        if self.augmentation is not None:
            image = self.augmentation.augment_random(image, num_augmentations=2)

        # 그레이스케일 변환 및 정규화
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 크기 조정 (32x128)
        image = cv2.resize(image, (128, 32))

        # 텐서 변환 및 정규화
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]

        # 레이블 인코딩
        label = [self.char_to_idx.get(char, 0) for char in text]
        label = torch.LongTensor(label)

        return image, label, len(text)


class CRNN(nn.Module):
    """CRNN 모델 (Convolutional Recurrent Neural Network)"""

    def __init__(self, num_classes, hidden_size=256):
        """
        CRNN 모델 초기화

        Args:
            num_classes: 클래스 수 (문자 집합 크기 + 1 for blank)
            hidden_size: LSTM 은닉 크기
        """
        super(CRNN, self).__init__()

        # CNN 특징 추출
        self.cnn = nn.Sequential(
            # Conv1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 16, 64]

            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [128, 8, 32]

            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # [256, 4, 32]

            # Conv5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Conv6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # [512, 2, 32]

            # Conv7
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU(inplace=True)  # [512, 1, 31]
        )

        # RNN (양방향 LSTM)
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True)

        # Fully Connected
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        순전파

        Args:
            x: 입력 이미지 [B, 1, H, W]

        Returns:
            로짓 [T, B, num_classes]
        """
        # CNN 특징 추출
        conv = self.cnn(x)  # [B, C, H', W']

        # [B, C, H', W'] -> [B, W', C*H']
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2).contiguous()  # [B, W', C, H']
        conv = conv.view(b, w, c * h)  # [B, W', C*H']

        # RNN
        output, _ = self.rnn(conv)  # [B, W', hidden*2]

        # FC
        output = self.fc(output)  # [B, W', num_classes]

        # [B, W', num_classes] -> [W', B, num_classes] (CTC 형식)
        output = output.permute(1, 0, 2)

        return output


def collate_fn(batch):
    """배치 콜레이트 함수"""
    images, labels, lengths = zip(*batch)

    # 이미지 스택
    images = torch.stack(images, 0)

    # 레이블 패딩
    max_len = max(lengths)
    padded_labels = []
    for label in labels:
        padded = torch.zeros(max_len, dtype=torch.long)
        padded[:len(label)] = label
        padded_labels.append(padded)

    labels = torch.stack(padded_labels, 0)
    lengths = torch.LongTensor(lengths)

    return images, labels, lengths


def train_epoch(model, dataloader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for images, labels, label_lengths in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # 순전파
        outputs = model(images)  # [T, B, num_classes]

        # CTC Loss
        input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
        loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, charset):
    """검증"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, label_lengths in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            # 순전파
            outputs = model(images)  # [T, B, num_classes]

            # CTC Loss
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
            total_loss += loss.item()

            # 정확도 계산 (greedy decoding)
            _, preds = outputs.max(2)  # [T, B]
            preds = preds.permute(1, 0)  # [B, T]

            for pred, label, length in zip(preds, labels, label_lengths):
                # CTC 디코딩 (중복 제거 및 blank 제거)
                pred_chars = []
                prev_char = None
                for p in pred:
                    if p != 0 and p != prev_char:  # 0은 blank
                        pred_chars.append(p.item())
                    prev_char = p

                # 레이블
                label_chars = label[:length].tolist()

                # 비교
                if pred_chars == label_chars:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    return total_loss / len(dataloader), accuracy


def main(args):
    """메인 학습 함수"""
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")

    # 문자 집합 구성
    charset = config.OCR_ALLOWED_CHARS
    num_classes = len(charset) + 1  # +1 for CTC blank

    print(f"문자 집합 크기: {len(charset)}")
    print(f"클래스 수: {num_classes}")

    # 데이터 증강
    augmentation = PlateAugmentation(seed=args.seed) if args.augment else None

    # 데이터셋
    train_dataset = PlateTextDataset(
        args.train_dir,
        args.train_labels,
        charset,
        augmentation=augmentation
    )

    val_dataset = PlateTextDataset(
        args.val_dir,
        args.val_labels,
        charset,
        augmentation=None  # 검증 시에는 증강 안 함
    )

    # 데이터로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # 모델
    model = CRNN(num_classes, hidden_size=args.hidden_size).to(device)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 손실 함수 및 옵티마이저
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 학습
    best_accuracy = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # 검증
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, charset)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 학습률 조정
        scheduler.step(val_loss)

        # 모델 저장
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'charset': charset
            }, args.save_path)
            print(f"✓ 모델 저장 (정확도: {val_accuracy:.4f})")

    print(f"\n학습 완료! 최고 정확도: {best_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="번호판 텍스트 인식 모델 학습")

    # 데이터 경로
    parser.add_argument('--train-dir', type=str, required=True, help="학습 이미지 디렉토리")
    parser.add_argument('--train-labels', type=str, required=True, help="학습 레이블 JSON 파일")
    parser.add_argument('--val-dir', type=str, required=True, help="검증 이미지 디렉토리")
    parser.add_argument('--val-labels', type=str, required=True, help="검증 레이블 JSON 파일")

    # 모델 설정
    parser.add_argument('--hidden-size', type=int, default=256, help="LSTM 은닉 크기")

    # 학습 설정
    parser.add_argument('--epochs', type=int, default=50, help="에폭 수")
    parser.add_argument('--batch-size', type=int, default=32, help="배치 크기")
    parser.add_argument('--lr', type=float, default=0.001, help="학습률")
    parser.add_argument('--gpu', action='store_true', help="GPU 사용")
    parser.add_argument('--num-workers', type=int, default=4, help="데이터로더 워커 수")
    parser.add_argument('--seed', type=int, default=42, help="랜덤 시드")

    # 증강
    parser.add_argument('--augment', action='store_true', help="데이터 증강 사용")

    # 저장 경로
    parser.add_argument('--save-path', type=str, default='models/crnn_best.pth', help="모델 저장 경로")

    args = parser.parse_args()

    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    main(args)
