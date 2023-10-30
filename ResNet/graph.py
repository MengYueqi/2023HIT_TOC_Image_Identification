import os
import matplotlib.pyplot as plt

# 结果保存在./result.txt文件内
# 绘制结果相关图像
train_loss = []
val_accuracy = []

# 读出train_loss
with open("./result.txt") as f:
    # 取出首行
    f.readline()
    l = f.readline().replace("[", "").replace("]", "").replace("\n", "")
    l = l.split(", ")
    train_loss = l
    # 将字符串类型转换为浮点类型
    for num in enumerate(train_loss):
        train_loss[num[0]] = float(num[1])

# 读出val_accuracy
with open("./result.txt") as f:
    # 取出前三行
    f.readline()
    f.readline()
    f.readline()
    l = f.readline().replace("[", "").replace("]", "").replace("\n", "")
    l = l.split(", ")
    val_accuracy = l
    # 将字符串类型转换为浮点类型
    for num in enumerate(val_accuracy):
        val_accuracy[num[0]] = float(num[1])


print(train_loss)
print(val_accuracy)

# 绘制 trainLoss 和 testAccu
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(train_loss, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Test Accuracy', color=color)
ax2.plot(val_accuracy, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()