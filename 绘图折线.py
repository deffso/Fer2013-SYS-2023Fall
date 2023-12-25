import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def process_and_plot(file_path, file_name):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析文件中的信息
    epochs = []
    train_loss = []
    val_loss = []
    test_loss = []
    train_acc = []
    val_acc = []
    test_acc = []

    for line in lines[1:]:  # 跳过标题行
        data = line.split()
        epochs.append(int(data[0]))
        train_loss.append(float(data[1]))
        train_acc.append(float(data[2]))
        val_loss.append(float(data[3]))
        val_acc.append(float(data[4]))
        test_loss.append(float(data[5]))
        test_acc.append(float(data[6]))
    
    special_epochs = epochs[::15]
    special_points = [val_acc[i] for i in range(0, len(val_acc), 15)]
    # 创建子图
    fig = make_subplots(rows=1, cols=1, subplot_titles=["Accuracy"])

    # 添加损失图
    #fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Training Loss'), row=1, col=1)
    #fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss'), row=1, col=1)
    #fig.add_trace(go.Scatter(x=epochs, y=test_loss, mode='lines+markers', name='Testing Loss'), row=1, col=1)
     # 更新布局
    
    # 添加准确率图
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Training Accuracy'))
    fig.add_trace(go.Scatter(x=special_epochs, y=special_points, mode='markers'))
    for epoch, point in zip(special_epochs, special_points):
        fig.add_annotation(x=epoch, y=point, text=f'{point:.2f}', showarrow=True, arrowhead=4, ax=0, ay=-40)    
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy'))
    fig.add_trace(go.Scatter(x=epochs, y=test_acc, mode='lines+markers', name='Testing Accuracy'))

    # 更新布局
    fig.update_layout(title_text=f'Training and Validation and Testing Accuracy - {file_name}', showlegend=True)
    
    # 保存图表为图片
    output_path = os.path.join(folder_path, file_name + '仅准确率_plotly.html')
    #output_path = os.path.join(folder_path, file_name + '仅准确率_plotly.jpg')
    fig.write_html(output_path)

    # 显示图表
    fig.show()

# 指定文件夹路径
folder_path = 'D:\\模式识别\\Fer2013-Facial-Emotion-Recognition-Pytorch-main\\ParrtenRegination\\汇报结果'

# 获取文件夹中的所有文件
file_names = os.listdir(folder_path)

# 遍历每个文件
for file_name in file_names:
    # 构建文件的完整路径
    file_path = os.path.join(folder_path, file_name)

    # 处理并绘制图表
    process_and_plot(file_path, file_name)
