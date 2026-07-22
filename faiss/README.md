# 🔎 Face Vector Search Examples

这是一个用于学习“人脸编码 + FAISS 精确向量搜索”的小型示例项目。项目包含两个结构相近的人脸搜索脚本，以及一个不涉及人脸模型的纯 FAISS 入门示例。两个主脚本采用线性教学结构，没有拆分成多层函数，便于按执行顺序阅读。

## 📚 官方项目链接

- [`face_recognition`](https://github.com/ageitgey/face_recognition)：基于 dlib 的 Python 人脸识别库。
- [`face_recognition` API 文档](https://face-recognition.readthedocs.io/en/latest/face_recognition.html)：查看 `face_locations()`、`face_encodings()`、`model="cnn"` 和 `model="large"` 等参数。
- [InsightFace](https://github.com/deepinsight/insightface)：人脸检测、对齐和识别工具箱。
- [InsightFace Python Package](https://github.com/deepinsight/insightface/tree/master/python-package)：`FaceAnalysis` Python 接口说明。
- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)：模型包和测试结果说明。
- [FAISS](https://github.com/facebookresearch/faiss)：高性能向量相似度搜索与聚类库。
- [FAISS 距离与余弦相似度说明](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)：官方说明如何使用归一化向量和内积实现余弦相似度。

## 📁 项目结构

```text
face_vector_search_examples/
├── faces/
│   ├── alice.jpg
│   ├── bob.jpg
│   └── charlie.jpg
├── query.jpg
├── face_search_face_recognition.py
├── face_search_insightface.py
├── faiss_vector_search_example.py
├── requirements_face_recognition.txt
├── requirements_insightface.txt
├── requirements_faiss.txt
└── README.md
```

`faces/` 用于存放人脸数据库图片。图片文件名去掉扩展名后，会作为搜索结果中显示的人名。`query.jpg` 是待查询的人脸图片。两个主脚本的文件顶部也重复写明了目录结构、图片放置位置和运行命令，单独打开脚本即可看到。

## 🧠 两个人脸版本的共同流程

两个脚本都按照“读取目录 → 编码数据库图片 → 归一化 → 建立索引 → 编码查询图片 → 搜索 → 降序打印”的顺序直接执行，并使用完全相同的 FAISS 搜索方式：

1. 从数据库图片中检测人脸。
2. 如果图片里有多张脸，选择面积最大的人脸。
3. 使用对应的人脸编码器生成特征向量。
4. 使用 `faiss.normalize_L2()` 对数据库向量和查询向量进行 L2 归一化。
5. 使用 `faiss.IndexFlatIP` 建立精确索引。
6. 对归一化向量执行内积搜索，此时内积等于余弦相似度。
7. 搜索全部数据库向量，并按相似度从高到低打印。

`IndexFlatIP` 是精确搜索，不使用 IVF、PQ 或 HNSW 等近似索引。它会将查询向量与索引中的全部向量比较，因此不会因为候选分区遗漏真正的最近邻。

## 📷 版本一：face_recognition

脚本：`face_search_face_recognition.py`

该版本使用：

- `face_recognition.face_locations(..., model="cnn")`：使用 CNN 人脸检测器。
- `face_recognition.face_encodings(..., model="large")`：使用 68 点人脸特征定位模型。
- dlib 人脸编码网络：输出 128 维人脸向量。
- `faiss.IndexFlatIP`：精确余弦相似度搜索。

⚠️ `model="large"` 指 68 点人脸特征定位模型，不代表生成另一种更大的编码网络。最终人脸编码仍然是 128 维。CNN 检测器在 CPU 环境中通常较慢。

安装：

```bash
pip install -r requirements_face_recognition.txt
```

运行：

```bash
python face_search_face_recognition.py
```

## 🧬 版本二：InsightFace

脚本：`face_search_insightface.py`

该版本使用：

- `FaceAnalysis(name="buffalo_l")`：加载 InsightFace 的 `buffalo_l` 模型包。
- `CPUExecutionProvider`：使用 ONNX Runtime CPU 推理。
- `face.embedding`：获取人脸识别向量，当前脚本会自动读取实际向量维度。
- `faiss.IndexFlatIP`：精确余弦相似度搜索。

安装：

```bash
pip install -r requirements_insightface.txt
```

运行：

```bash
python face_search_insightface.py
```

首次运行时，InsightFace 可能自动下载 `buffalo_l` 模型文件。模型文件通常保存在用户目录下的 `.insightface/models/` 中。

⚠️ InsightFace 代码仓库采用开源许可证，但官方提供的预训练模型通常限定为非商业研究用途。商业项目应确认模型授权或使用自有授权模型。

## 🧮 纯 FAISS 最小示例

脚本：`faiss_vector_search_example.py`

该脚本只使用 NumPy 和 `faiss-cpu`，不使用任何人脸库。它通过几个二维向量演示：

- `IndexFlatL2`：平方 L2 距离，数值越小越相似。
- `IndexFlatIP`：向量归一化后的内积，即余弦相似度，数值越大越相似。
- `index.add()`：向索引添加数据库向量。
- `index.search()`：搜索最相似的前 K 个向量。
- FAISS 返回的距离、相似度和向量编号如何对应到名称。

安装：

```bash
pip install -r requirements_faiss.txt
```

运行：

```bash
python faiss_vector_search_example.py
```

## 📊 如何理解搜索分数

### 余弦相似度

两个向量先经过 L2 归一化后：

```python
faiss.normalize_L2(database_vectors)
faiss.normalize_L2(query_vector)
index = faiss.IndexFlatIP(vector_dimension)
```

此时 `IndexFlatIP` 返回的内积等于余弦相似度。分数越大，向量方向越接近。

### L2 距离

`IndexFlatL2` 返回平方 L2 距离，不执行最后的开平方。数值越小，两个向量越接近。

对于已经归一化的向量，平方 L2 距离和余弦相似度满足：

```text
squared_L2 = 2 - 2 × cosine_similarity
```

因此二者通常会产生相同的最近邻排序，但返回分数的含义和方向不同。

## ⚠️ 重要说明

- 相似度不是概率。`0.85` 不表示“85% 确定是同一个人”。
- 同一个模型在不同摄像头、光照、姿态、年龄和图像质量下，分数分布可能不同。
- 正式系统应准备“同一个人”和“不同人”的验证数据，再确定业务阈值。
- 一张数据库图片最好只包含一张清晰、正面的人脸。
- 两种编码器生成的向量不能混在同一个索引中。128 维 `face_recognition` 向量与 InsightFace 向量属于不同特征空间，不能直接比较。
- 本项目用于学习向量搜索，不包含活体检测、防照片攻击、权限管理、数据库持久化或隐私合规功能。
- 人脸向量属于敏感生物识别数据。实际应用中应执行加密、访问控制、最小化保存和合规评估。

## ✅ 建议的研究顺序

1. 先运行 `faiss_vector_search_example.py`，理解 L2 距离和余弦相似度。
2. 再运行 `face_search_face_recognition.py`，观察 128 维编码结果。
3. 最后运行 `face_search_insightface.py`，比较两个编码器的搜索分数。
4. 使用多张同人照片和不同人照片收集分数，研究阈值，而不是直接照搬固定阈值。
