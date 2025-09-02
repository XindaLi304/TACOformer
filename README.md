# TACOformer
Official Pytorch implementation of token-channel compounded cross attention for multimodal emotion recognition


data_preprocess: 
  the processed shape in "data_exg.np" file should be [1280,60,n_channels,128].
  The details of the preprocessing are shown in https://arxiv.org/pdf/2306.13592
  
tacoformer/
├─ main.py                  # 入口：数据加载 -> 划分并保存测试集 -> 超参搜索 -> 训练最佳模型 -> 测试集评估
├─ config.py                # 路径与超参网格配置
├─ data.py                  # 数据加载、拼接、划分、DataLoader 构造
├─ model.py                 # 你的 ViT + TACO Cross-Attention 模型（功能不改、注释英文）
├─ train.py                 # 训练与评估循环（保持你的损失/精度计算逻辑）
├─ search.py                # k-fold 超参搜索
├─ utils.py                 # 工具函数（设随机种、度量、保存等）
└─ requirements.txt         # 依赖
