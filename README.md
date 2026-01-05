# SBERT-replication

本项目基于论文 Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks，在 Jittor 框架下实现了 SBERT 模型，并复现了论文中关于无监督 STS 评估和消融实验的结果。已训练好的模型参数可通过 https://cloud.tsinghua.edu.cn/d/ee88e0dc4f9a421fa1b8/ 下载。

data_utils.py 和 dataloader.py 用于加载 NLI 数据集，evaluate.py 用于训练阶段校验，experiments/test_NLI.py 用于在 NLI 数据集上测试，train.py 用于训练。models 下 SBERT.py 包含强行转换 Pytorch 参数、Jittor 版 BERT 解码句子和池化层，modeling_jittor.py 是基于 Jittor 实现的 transformer 层，siamese_network.py 实现了孪生网络结构（默认使用分类目标函数，句子向量连接方式为 $(u,v,|u-v|)$）。
STSdatasets_loader.py、run_unsupSTS.py 和 model_loader.py 用于在 STS 上进行测试。
ablation_experiments 下是消融实验的代码。config.py 是基础配置设定，其余三份代码分别负责池化策略、目标函数和句子向量连接方式的消融实验。
