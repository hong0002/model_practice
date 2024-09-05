from efficientnet.model_d import EfficientNet
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10  # 예시 데이터셋으로 CIFAR-10 사용
from tqdm import tqdm  # tqdm 임포트
import numpy as np

def train_model():
    # 1. 데이터 준비
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet은 224x224 이미지를 입력으로 받습니다.
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 데이터셋 로드
    full_train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 전체 데이터셋에서 10%만 사용하도록 샘플링
    train_indices = np.random.choice(len(full_train_dataset), len(full_train_dataset) // 10, replace=False)
    test_indices = np.random.choice(len(full_test_dataset), len(full_test_dataset) // 10, replace=False)

    # Subset으로 부분 데이터셋 생성
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 2. 모델 정의
    model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=10)  # CIFAR-10은 10개의 클래스를 가짐
    model.train()  # 모델을 학습 모드로 전환

    # 3. 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()  # 분류 문제에 사용되는 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저 사용

    # 4. 학습 루프 정의
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU가 있으면 사용
    model = model.to(device)  # 모델을 GPU로 이동

    num_epochs = 5  # 학습할 epoch 수
    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정
        running_loss = 0.0

        # tqdm을 사용하여 프로그래스 바 생성
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 GPU로 이동

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 역전파 및 옵티마이저 스텝
            loss.backward()
            optimizer.step()

            # 손실 출력
            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / (train_bar.n + 1))

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # 검증 루프 (옵션)
        model.eval()  # 모델을 평가 모드로 전환
        correct = 0
        total = 0
        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation")  # tqdm 프로그래스 바 생성
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix(accuracy=100 * correct / total)

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    print("Training Finished!")

if __name__ == '__main__':
    train_model()
