import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
# from torchvision import transforms
from pre import FetalDataset, compute_miou, plot_metrics, compute_dice, visualize_and_save_predictions
from torch.utils.tensorboard import SummaryWriter
from unet import UNet 
from tqdm import tqdm

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def label_to_rgb(tensor, num_classes):
            cmap = torch.tensor([
                [0, 0, 0],      # 类别0: 黑色
                [255, 0, 0],    # 类别1: 红色
                [0, 255, 0]     # 类别2: 绿色
            ], dtype=torch.uint8)
            return cmap[tensor.long()]


def train(model, train_loader, criterion, optimizer, device, num_classes, pbar=None):
    model.train()
    running_loss = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_batches = len(train_loader)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)                # 获取每个像素的预测类别
        miou = compute_miou(preds, labels, num_classes)
        dice = compute_dice(preds, labels, num_classes)

        running_loss += loss.item()
        total_miou += miou
        total_dice += dice

        if pbar is not None:
            pbar.update(1)  # 更新进度条
            pbar.set_postfix({"Loss": loss.item(), "mIoU": miou, "dice": dice})

    avg_loss = running_loss / total_batches
    avg_miou = total_miou / total_batches
    avg_dice = total_dice / total_batches
    return avg_loss, avg_miou, avg_dice


def validate(model, val_loader, criterion, device, num_classes, epoch, pbar, writer=None):
    model.eval()  
    total_miou = 0.0
    total_dice = 0.0
    total_loss = 0.0
    total_batches = len(val_loader)
    sample_images = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)

            inputs, labels = inputs.to(device), labels.to(device)
            # print(f"Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
            # print(f"Label unique values: {torch.unique(labels)}")

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)                     # 获取每个像素的预测类别

            if len(preds.shape) == 2:
                preds = preds.unsqueeze(0)
            if len(labels.shape) == 2:
                labels = labels.unsqueeze(0)

            miou = compute_miou(preds, labels, num_classes)
            dice = compute_dice(preds, labels, num_classes)
            total_miou += miou
            total_dice += dice

            if batch_idx == len(val_loader) - 1:
                sample_images.append((
                    inputs.detach().cpu(),
                    labels.detach().cpu(),
                    preds.detach().cpu()
                ))

        if pbar is not None:
            pbar.update(1)  # 更新进度条
            pbar.set_postfix({"Loss": loss.item(), "mIoU": miou, "dice": dice})

    avg_miou = total_miou / total_batches
    avg_loss = total_loss / total_batches
    avg_dice = total_dice / total_batches

    if writer is not None and sample_images:
        inputs, labels, preds = sample_images[0]
        for i in range(min(4, inputs.shape[0])):
            img_tensor = inputs[i]
            img = (img_tensor.permute(1, 2, 0) * 255).byte()

            label_rgb = label_to_rgb(labels[i], num_classes)
            pred_rgb = label_to_rgb(preds[i], num_classes)

            label_rgb = label_rgb.permute(2, 0, 1).float() / 255
            pred_rgb = pred_rgb.permute(2, 0, 1).float() / 255
            
            
            combined = torch.cat([img_tensor, label_rgb, pred_rgb], dim=-1)
            writer.add_image(f'Val/Compare_{i}', combined, epoch)

    
    return avg_miou, avg_dice, avg_loss

def main():
    batch_size = 16
    num_epochs = 3000
    learning_rate = 1e-5
    best_miou = 0.6
    best_dice = 0.75
    checkpoint_dir = "./checkpoints/epoch{}bs{}".format(num_epochs, batch_size)
    image_dir = './datafetal/labeled_data/images'  
    label_dir = './datafetal/labeled_data/labels'                 # label_exchange
    val_image_dir = './datafetal/labeled_valdata/images'  
    val_label_dir = './datafetal/labeled_valdata/labels'   # labels_change
    num_classes = 3
    output_dir = './visualizations/valdata_new'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir='./logs/epoch{}bs{}'.format(num_epochs, batch_size))

    image_transform = T.Compose([
        T.Resize((512, 512)),  
        T.ToTensor()
          
    ])
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    label_transform = T.Compose([
        T.Resize((512, 512), 
            interpolation=T.InterpolationMode.NEAREST),  
        lambda x: torch.from_numpy(np.array(x)).long()
    ])

    train_dataset = FetalDataset(image_dir, label_dir, image_transform=image_transform, label_transform=label_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = FetalDataset(val_image_dir, val_label_dir, image_transform=image_transform, label_transform=label_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_chns=3, class_num=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss() 

    train_losses = []
    train_mious = []
    val_mious = []
    train_dices = []
    val_dices = []
    epochs_list = []

    for epoch in range(num_epochs):
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch", disable=(epoch + 1) % 10 != 0) as pbar:
            train_loss, train_miou, train_dice = train(model, train_loader, criterion, optimizer, device, num_classes, pbar)

        if (epoch + 1) % 50 == 0:
            print(f"training :After Epoch {epoch+1}: Loss: {train_loss:.5f}, train mIoU: {train_miou:.5f}, train dice: {train_dice:.5f}")

        # 验证阶段
        with tqdm(total=len(val_loader), desc=f"Validating Epoch {epoch+1}/{num_epochs}", unit="batch", disable=(epoch + 1) % 10 != 0) as pbar:
            val_miou, val_dice, val_loss = validate(model, val_loader, criterion, device, num_classes, epoch, pbar, writer)
            print(f"Validating Epoch {epoch+1}, val mIoU is:{val_miou:.5f}, val dice is:{val_dice:.5f}, val loss is:{val_loss:.5f}")
            writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            
            
            writer.add_scalars('Dice', {
                'train': train_dice,
                'val': val_dice
            }, epoch)
    

        train_losses.append(train_loss)
        train_mious.append(train_miou)
        val_mious.append(val_miou)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        epochs_list.append(epoch + 1)
        # visualize_and_save_predictions(val_loader, model, device, output_dir, num_classes, epoch)
        # if (epoch + 1) == num_epochs:
        #     plot_metrics(epochs_list, train_losses, train_mious, val_mious, train_dices, val_dices)

        # 保存最佳模型
        if val_miou > best_miou and val_dice > best_dice:
            best_miou = val_miou
            best_dice = val_dice
            model_path = os.path.join(checkpoint_dir, "best_{}_valdice{}.pth".format(epoch, val_dice))  
            torch.save(model.state_dict(), model_path)
            print(f"==========Saved best model with mIoU: {best_miou:.4f}, Dice: {best_dice:.4f}============")    
    
    writer.close()
    print(f"日志保存在{writer.log_dir}")

if __name__ == "__main__":
    main()


# Saved best model with mIoU: 0.8649, Dice: 0.9465
# Saved best model with mIoU: 0.8802, Dice: 0.9526
# Epoch 1700: Loss: 0.03086, train mIoU: 0.95763, train dice: 0.98383
# Epoch 2350: Loss: 0.02336, train mIoU: 0.95912, train dice: 0.98454