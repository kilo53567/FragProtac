# =============================================================================
# FragProtac
# 本模型用于PROTAC分子二分类活性预测，建模弹头、连接子、E3配体三者之间的三元相互作用，
# 并融合通路嵌入特征。本仓库可复现我们论文的相关实验。
# 论文在投，录用后补充引用信息。
#
# 训练流程：
# 1. 准备数据
#    脚本需要弹头SMILES、连接子SMILES、E3配体SMILES、预计算通路嵌入向量以及二分类活性标签。
#    所有信息存放在同一个CSV文件中。下文给出CSV格式规范。
#    如果使用自有数据集，请修改脚本里的文件路径。
#
# 2. 核对CSV格式
#    输入CSV必须包含以下列：
#    label, warhead smiles, linker smiles, e3 ligand smiles, Pathway_Combined_128d
#    为保证代码鲁棒性，缺失的SMILES会自动替换为C；
#    缺失的通路向量会用零向量填充。
#
# 3. 配置运行环境
#    可以使用下面命令搭建conda环境。本模型在支持CUDA的NVIDIA显卡上训练和测试。
#
#        conda create -n fragprotac python=3.9
#        conda activate fragprotac
#        pip install torch torchvision torchaudio
#        pip install torch_geometric
#        pip install rdkit-pypi pandas numpy scikit-learn matplotlib seaborn
#
#    建议根据你的CUDA驱动版本匹配PyTorch与PyG版本。
#
# 4. 修改配置参数
#    打开训练脚本，按需修改全局路径与超参：
#        CSV_PATH = "/FragProtac/data/PROTAC-fine/protac-fine.csv"
#        OUTPUT_DIR = "/FragProtac"
#
# 5. 运行训练脚本
#        python FragProtac_train.py
# =============================================================================

