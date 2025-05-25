import matplotlib.pyplot as plt

# 数据
categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral']
counts = [1271, 1271, 1271, 1271, 1271, 1087]

# 使用默认字体，设置大小为12（小四号字号）
plt.rcParams.update({'font.size': 12})

# 绘图
plt.figure(figsize=(8, 5))
bars = plt.bar(categories, counts, color='skyblue')

# 添加标签和标题
plt.title("CREMA-D Dataset")
plt.xlabel("Categories")
plt.ylabel("Sample Counts")
plt.xticks(rotation=45)  # 让类别标签斜着显示
plt.yticks()

# 移除顶部和右侧边框线
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 在每个柱子上显示数量
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(count),
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
