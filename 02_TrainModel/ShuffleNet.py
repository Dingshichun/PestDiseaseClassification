
import os # 导入操作系统模块
import argparse
import torch
import torchvision
import torch.optim as optim #优化函数
from torch.utils.tensorboard import SummaryWriter #tensorboard可视化工具
from torchvision import transforms #数据增强
import torch.optim.lr_scheduler as lr_scheduler #学习率设置函数

# MyDataSet和utils参考my_dataset.py和utils.py，
from my_dataset import MyDataSet # 从my_dataset导入MyDataSet函数，可返回图像和其标签。
# utils是之前定义过的模块，从中导入一些函数和类。
from utils import read_split_data, train_one_epoch, evaluate # 文件夹标签划分，训练一轮的函数，一个验证函数
from utils import freeze_model_layers # 冻结输出层以外的其它层

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(args) # 打印导入的参数。
    print('Start Tensorboard with "tensorboard --logdir=runs",view at http://localhost:6006/')
    tb_writer = SummaryWriter(args.rundata_path) # 括号内设置数据可视化路径,默认在当前文件路径创建runs文件夹
    # 保存模型的目录如果不存在，就创建
    if os.path.exists(args.save_path) is False:
        os.makedirs(args.save_path)
    # read_split_data()用于划分训练和验证
    train_images_path,train_images_label,val_images_path,val_images_label \
        = read_split_data(args.data_path)

    # 数据增强方法
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),  # 改变原图大小
            transforms.RandomRotation(90), # 随机旋转0~90度
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomVerticalFlip(),  # 随机垂直翻转
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化
        ]),
        "val": transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),  # 改变原图大小
            transforms.ToTensor(),  # 转为张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
        ])
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(
        images_path = train_images_path,
        images_class = train_images_label,
        transform = data_transform["train"]
    )
    # 实例化验证数据集
    val_dataset = MyDataSet(
        images_path = val_images_path,
        images_class = val_images_label,
        transform = data_transform["val"]
    )
    batch_size = args.batch_size
    # nw是加载图像的进程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    # 加载训练数据
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True, #随机打乱
        pin_memory = True,
        # pinned memory，锁页内存，
        # 设置为true时，内存中的tensor转移到GPU上会更快
        num_workers = nw,
        collate_fn = train_dataset.collate_fn  # collate_fn是将数据整理为一个批次的函数。
    )
    # 加载验证数据
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False, # 验证时没必要将数据打乱。
        pin_memory = True,
        num_workers = nw,
        collate_fn = val_dataset.collate_fn
    )
    # 加载模型
    model = torchvision.models.shufflenet_v2_x2_0(weights=torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
    in_feature=model.fc.in_features
    model.fc=torch.nn.Linear(in_feature,out_features=args.pestdisease_nums,bias=True)
    # print(model)
    # 冻结输出层以外的层
    freeze_model_layers(model,args.freeze_layers)
    # 模型转移到训练使用的 device，一般是 gpu，单 gpu 一般是 cuda:0
    model = model.to(device)
    
    # 获取需要训练的参数列表，即梯度为 true 的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    for para in pg:
        print(f"training parameter: {para}")
        
    # 使用带动量的随机梯度下降优化函数
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    # 动态更改学习率。每 step_size 个 epoch 之后，学习率乘以 gamma
    scheduler=lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    
    # 在每一个epoch后，如果当前准确率超过了历史最大准确率，就保存模型。
    # 这样，保存的是历史最佳模型。
    # 早停机制是：在连续若干个 epoch（比如 11 个）都没有提高时，就停止训练。
    val_acc_list = []
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 11  # 连续 11 个 epoch 准确率不提高就停止
    best_model_state = None  # 保存最佳模型权重

    for epoch in range(args.epochs):
        # 训练
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        scheduler.step()
        
        # 验证
        acc = evaluate(model=model,
                    data_loader=val_loader,
                    device=device)
        
        print(f"[epoch {epoch}] accuracy: {round(acc, 3)}")
        
        # TensorBoard 记录
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        
        val_acc_list.append(acc)
        
        # 检查是否是最佳准确率
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            patience_counter = 0  # 重置计数器
            
            # 保存最佳模型权重
            best_model_state = model.state_dict().copy()
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'val_acc': acc,
            }
            torch.save(checkpoint, 
                    f"{args.modelsave_path}best_checkpoint_epoch{epoch}_acc{acc:.3f}.pth")
            
            print(f"新最佳准确率: {acc:.3f} (Epoch {epoch})")
        else:
            patience_counter += 1
            print(f"准确率未提高，耐心计数: {patience_counter}/{patience}")
        
        # 检查早停条件
        if patience_counter >= patience:
            print(f"\n早停触发! 连续 {patience} 个 epoch 准确率未提高")
            print(f"最佳准确率: {best_acc:.3f} (Epoch {best_epoch})")
            
            # 保存最终的最佳模型
            if best_model_state is not None:
                # 恢复最佳模型
                model.load_state_dict(best_model_state)
                
                # 保存模型参数
                torch.save(model.state_dict(),
                        f"{args.modelsave_path}best_state_dict.pth")
                # 保存完整模型
                torch.save(model,
                        f"{args.modelsave_path}best_complete.pth")
                
                # 保存最佳检查点
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_state,
                    'best_acc': best_acc,
                    'val_acc_history': val_acc_list,
                }, f"{args.modelsave_path}best_model_summary.pth")
            
            break

    # 关闭TensorBoard写入器（在循环外关闭）
    tb_writer.close()

    # 如果正常完成所有epoch
    if patience_counter < patience:
        print(f"完成所有 {args.epochs} 个 epoch 的训练")
        print(f"最佳准确率: {best_acc:.3f} (Epoch {best_epoch})")
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 种类数修改为自己数据集的种类数。
    parser.add_argument('--pestdisease_nums',type=int,default=38,help="病虫害类别数")
    parser.add_argument('--epochs',type=int,default=100,help="训练轮次")
    parser.add_argument('--batch_size',type=int,default=28,help="批次大小")
    parser.add_argument('--lr',type=float,default=0.01,help="learning rate")
    parser.add_argument('--image_size',type=int,default=224,help='input image size')

    # 数据路径
    parser.add_argument('--data_path',type=str,
                        default="./color_dataset/TrainVal",
                        help="the path of image data")
    # tensorboard 数据存储路径
    parser.add_argument('--rundata_path',type=str,
                        default='./RunData/ShuffleNet',
                        help="save the train result data")
    # 模型保存路径，用于存储模型，已经包含了模型命名的部分前缀
    parser.add_argument('--modelsave_path',type=str,
                        default='./ModelSave/ShuffleNet/ShuffleNet_',help="save the model")
    # 模型存储路径，用于确定存储路径是否存在，不包含模型命名的前缀
    parser.add_argument('--save_path',type=str,
                        default='./ModelSave/ShuffleNet',help="save the model")

    parser.add_argument('--freeze_layers',type=bool,default=True,
                        help="是否冻结输出层以外的层")
    parser.add_argument('--device',default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    
    
    args = parser.parse_args(args=[]) # 不传入，使用默认参数。
    main(args) # 调用定义的main()函数。

