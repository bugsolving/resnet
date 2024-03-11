# from model import ResNet18
# import torch
# import torch.nn as nn
# from validate import validate
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from writer import MyWriter

# print("Starting script...")  # 打印调试信息

# def train(train_loader, test_loader, writer, epochs, lr, device):
#     best_l = 1000
#     model = ResNet18().to(device)
#     print("Model created and moved to device.")  # 打印模型创建信息

#     # 使用Adam优化器，学习率设置为0.001
#     optimizer_e = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()
#     # 设置训练轮数为20
#     for epoch in range(20):
#         model.train()
#         train_rmes, train_mae, train_loss = 0., 0., 0.
#         step = 0
#         loader = tqdm(train_loader)
#         for img, label in loader:
#             img, label = img.to(device), label.to(device).to(torch.float32)
#             print(f"Image shape: {img.shape}, Label shape: {label.shape}")  # 打印图像和标签的形状
#             optimizer_e.zero_grad()
#             score = model(img)
#             print(f"Score shape: {score.shape}")  # 打印模型输出的形状
#             label = label.view(-1, 1)  # 添加这行代码
#             loss = criterion(score, label)
#             train_loss += loss.item()
#             rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
#             train_rmes += rmse
#             mae = torch.abs(score - label).mean().item()
#             train_mae += mae
#             loss.backward()
#             optimizer_e.step()
#             step += 1
#             loader.set_description("Epoch:{} Step:{} RMSE:{:.2f} MAE:{:.2f}".format(epoch, step, rmse, mae))
#         train_rmes /= step
#         train_mae /= step
#         train_loss /= step
#         model.eval()
#         val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion)
#         writer.log_train(train_rmes, train_mae, train_loss, val_rmes, val_mae, val_loss, epoch)
#         if val_loss < best_l:
#             torch.save({'ResNet': model.state_dict()}, './ResNet.pth')
#             print('Save model!,Loss Improve:{:.2f}'.format(best_l - val_loss))
#             best_l = val_loss
#         print('Train RMSE:{:.2f} MAE:{:.2f} \t Val RMSE:{:.2f} MAE:{:.2f}'.format(train_rmes, train_mae, val_rmes,
#                                                                                    val_mae))

# print("Script finished.")  # 打印脚本完成信息

# # 创建数据加载器
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# train_dataset = datasets.ImageFolder(root='/root/resnet18/ResNet18/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_dataset = datasets.ImageFolder(root='/root/resnet18/ResNet18/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # 创建日志记录器
# writer = MyWriter('./logs')

# # 设置训练轮数
# epochs = 20

# # 设置学习率
# lr = 0.001

# # 设置设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 调用train函数
# train(train_loader, test_loader, writer, epochs, lr, device)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# from model import ResNet18
# import torch
# import torch.nn as nn
# from validate import validate
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from writer import MyWriter
# import torch.optim.lr_scheduler as lr_scheduler

# print("Starting script...")

# def train(train_loader, test_loader, writer, epochs, lr, device):
#     best_l = 1000
#     model = ResNet18().to(device)
#     print("Model created and moved to device.")

#     optimizer_e = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#     scheduler = lr_scheduler.StepLR(optimizer_e, step_size=10, gamma=0.1)  # 添加学习率调度器

#     for epoch in range(epochs):
#         model.train()
#         train_rmes, train_mae, train_loss = 0., 0., 0.
#         step = 0
#         loader = tqdm(train_loader)
#         for img, label in loader:
#             img, label = img.to(device), label.to(device).to(torch.float32)
#             optimizer_e.zero_grad()
#             score = model(img)
#             label = label.view(-1, 1)
#             loss = criterion(score, label)
#             train_loss += loss.item()
#             rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
#             train_rmes += rmse
#             mae = torch.abs(score - label).mean().item()
#             train_mae += mae
#             loss.backward()
#             optimizer_e.step()
#             step += 1
#             loader.set_description(f"Epoch:{epoch} Step:{step} RMSE:{rmse:.2f} MAE:{mae:.2f}")
#         scheduler.step()  # 更新学习率
#         train_rmes /= step
#         train_mae /= step
#         train_loss /= step
#         model.eval()
#         val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion)
#         writer.log_train(train_rmes, train_mae, train_loss, val_rmes, val_mae, val_loss, epoch)
#         if val_loss < best_l:
#             torch.save({'ResNet': model.state_dict()}, './ResNet.pth')
#             print(f'Save model! Loss Improved: {best_l - val_loss:.2f}')
#             best_l = val_loss
#         print(f'Train RMSE:{train_rmes:.2f} MAE:{train_mae:.2f} \t Val RMSE:{val_rmes:.2f} MAE:{val_mae:.2f}')

# print("Script finished.")

# # 数据预处理，添加数据增强
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),  # 数据增强
#     transforms.RandomRotation(10),  # 数据增强
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# train_dataset = datasets.ImageFolder(root='/root/resnet18/ResNet18/train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_dataset = datasets.ImageFolder(root='/root/resnet18/ResNet18/test', transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# writer = MyWriter('./logs')
# epochs = 20
# lr = 0.001
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train(train_loader, test_loader, writer, epochs, lr, device)
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
from model import ResNet18
import torch
import torch.nn as nn
from validate import validate
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter  # 使用tensorboardX库
import torch.optim.lr_scheduler as lr_scheduler

print("Starting script...")

def train(train_loader, test_loader, writer, epochs, lr, device):
    best_l = 1000
    model = ResNet18().to(device)
    print("Model created and moved to device.")

    optimizer_e = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = lr_scheduler.StepLR(optimizer_e, step_size=10, gamma=0.1)  # 添加学习率调度器

    for epoch in range(epochs):
        model.train()
        train_rmes, train_mae, train_loss = 0., 0., 0.
        step = 0
        loader = tqdm(train_loader)
        for img, label in loader:
            img, label = img.to(device), label.to(device).to(torch.float32)
            optimizer_e.zero_grad()
            score = model(img)
            label = label.view(-1, 1)
            loss = criterion(score, label)
            train_loss += loss.item()
            rmse = torch.sqrt(torch.pow(torch.abs(score - label), 2).mean()).item()
            train_rmes += rmse
            mae = torch.abs(score - label).mean().item()
            train_mae += mae
            loss.backward()
            optimizer_e.step()
            step += 1
            loader.set_description(f"Epoch:{epoch} Step:{step} RMSE:{rmse:.2f} MAE:{mae:.2f}")
            writer.add_scalar('Train/Loss', loss.item(), step)  # 记录训练损失
            writer.add_scalar('Train/RMSE', rmse, step)  # 记录训练RMSE
            writer.add_scalar('Train/MAE', mae, step)  # 记录训练MAE
        scheduler.step()  # 更新学习率
        train_rmes /= step
        train_mae /= step
        train_loss /= step
        model.eval()
        val_rmes, val_mae, val_loss = validate(model, test_loader, device, criterion)
        writer.add_scalar('Val/Loss', val_loss, step)  # 记录验证损失
        writer.add_scalar('Val/RMSE', val_rmes, step)  # 记录验证RMSE
        writer.add_scalar('Val/MAE', val_mae, step)  # 记录验证MAE
        if val_loss < best_l:
            torch.save({'ResNet': model.state_dict()}, './ResNet.pth')
            print(f'Save model! Loss Improved: {best_l - val_loss:.2f}')
            best_l = val_loss
        print(f'Train RMSE:{train_rmes:.2f} MAE:{train_mae:.2f} \t Val RMSE:{val_rmes:.2f} MAE:{val_mae:.2f}')

print("Script finished.")

# 数据预处理，添加数据增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 数据增强
    transforms.RandomRotation(10),  # 数据增强
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root='/root/resnet18/ResNet18/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.ImageFolder(root='/root/resnet18/ResNet18/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

writer = SummaryWriter('./logs')  # 使用tensorboardX的SummaryWriter
epochs = 20
lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train(train_loader, test_loader, writer, epochs, lr, device)