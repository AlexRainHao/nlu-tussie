NER/Classifier baseline使用
===============

Release
-------

- `pre_1.0.0`

# 目录结构
1. [1.概述](#Intro)
2. [2.模型训练](#Train)
3. [3.Docker服务](#Docker)
4. [4.试用服务](#TruIt)
5. [5.未来改进及计划](#Future)
6. [6.Bug反馈及改进提交](#Bugs)

---
<br>

## 1.概述

### 1.1 项目简介

一项支持
    * NER
    * 分类器
多项 `baseline` 模型的 **训练和测试** 快速使用
训练采用 `Pipeline` 方式 *自顶向下* 进行

除此之外，提供基于 `Dockerfile` 的 镜像构建 和 部署调用

### 1.2 项目参考

项目参考 `rasa-nlu==0.12.3` 沿用和修改
[Rasa](https://rasa.com/)
---
<br>

## 2.模型训练

### 2.1 pipeline
#### 2.1.1 简介

由 配置文件(`yml`) 给定所需的各个

* 预处理
* 特征提取
* 模型训练

等 `sub-pipe名称` 以及 `各自对应的参数`, 在训练和模型调用，也均将对每个 `sub-pipe` 执行 *自顶向下* 的方式使用

每个 `sub-pipe` 在调用后根据自身定义将提供相应的 `信息流`，供之后的 pipe 进行使用，其中常用的 `信息流` 以及 对应的 `变量名` 包括

* 分词结果 --> tokens
* 词性 --> pos
* 引用实体 --> preEntity（主要是 `spacy` 和 `lac` 使用）
* 文本向量 --> text_features
* 预测实体 --> entities
* 分类结果 --> intent

#### 2.1.2 pipeline 分类

在项目路径下，主要存在以下预设 pipe

| 文件夹名 | 功能 | 备注 |
| --- | --- | --- |
| tokenizers | 提供分词、词性、引用实体 | 使用 `spacy` 需提前使用 `spacy_nlp` pipe（见第2.2.1章和2.3.1章）|
| featurizers | 提供 词、句 向量 | |
extractors | ner 模型 | |
classifiers | 分类器 | |

#### 2.1.3 pipeline 注册

所有 `sub-pipe` 将注册到 `registry.py` 才会被配置文件读取到

### 2.2 模型训练和测试

#### 2.2.1 配置文件

提供几个 `示例配置yml` 进行训练

进入 `doc` 目录

```bash
cd doc
```

目录下的 `config` 文件夹下提供了 **5个NER** 的配置

* config_bertcrf.yml --> 基于bert-bilstm-crf的NER
* config_lstmcrf --> 基于 bilstm-crf的NER
* config_crfpp --> 基于统计规则的线性链crf的NER
* config_spacy --> 使用 *spacy* 预训练模型的NER
* config_normal --> 远古版本通用型NER的配置文件，提供18种实体输出（可能不适配现版本，供相关配置参考）

#### 2.2.2 训练脚本

该版本暂时提供脚本训练

修改 `train_nlu.py` 文件的 配置路径

```python
config = './config/config_bertcrf.yml'
```

然后进入虚拟环境，建议使用210的环境，或者在 `build_deped/requirements.txt` 进行安装

```python
source /home/user/yuanyh/vitualEnv/nerenv/bin/activate

python3 train_nlu.py
```

#### 2.2.3 输出结果

在制定的输出结果中，将按照各 `sub-pipe` 定义要求保存模型，以及对应的模型验证结果（已对部分模型做适配），如 `crfpp` 下存在 `eval_ner_crf.txt` 文件，保存如下结果

```python
*******Epoch: 50
             precision    recall        f1

      total      0.860     0.800     0.748          
     number      1.000     0.667     0.800          
       case      0.923     0.923     0.923          
     people      1.000     1.000     1.000          
   progress      0.750     0.789     0.769          
        adv      0.688     0.688     0.688          
responsible      1.000     0.714     0.833          
      court      1.000     0.667     0.800          
    conduct      0.750     0.375     0.500          
    hanover      1.000     1.000     1.000          

```

#### 2.2.4 调用测试模型

针对离线测试，该版本提供 `脚本测试` 进行 `命令行测试`

```
python3 test_nlu.py
```

### 2.3 部分pipe参数说明

#### 2.3.1 spacy

##### 2.3.1.1 分词

使用spacy进行分词，需使用如下配置
其中 `model` 是 spacy 预训练模型 `zh_core_web_lg:v2.3.0`

```yml
- name: "nlp_spacy"
  model: "/home/user/yuanyh/ner_dev/model/CoreModel"
- name: "tokenizer_spacy"
```

##### 2.3.1.2 模型预测

spacy的模型预测对应pipe为 `spacy_entity_extractor`
提供了一系列后处理参数，现进行简要说明

```python
"interest_entities": {
    "RESIDENT": ["FAC", "GPE", "LOC"]
}

"confidence_threshold": 0.7

"patterns": {
    "RESIDENT": [["[门幢楼栋室巷屯乡镇元层区庄址村]$", "conj"]],
    "PER": [["[\da-zA-Z]", "clear"]]
}
```

| 参数名 | 功能 |
| --- | --- |
| interest_entities | 将模型的 value 映射为 key 进行输出 |
| confidence_threshold | 按照置信度过滤 |
| patterns | 以正则方式对连续的 key 实体进行拼接(conj)或过滤(clear)|

#### 2.3.2 lac

##### 2.3.2.1 模型预测

lac的模型预测对应pipe为 `lac_entity_extractor`
同样以 spacy 方式提供后处理参数

#### 2.3.3 bilstm_crf类

##### 2.3.3.1 模型配置

该 pipe 提供 **4种** 基于 nn-crf 的模型，包括

* embedding-crf
* embedding-bilstm-crf
* bert-crf
* bert-bilstm-crf


通过以下4个参数进行相关设置，其中 `embedding` 参数 可选 **["embedding", "bert"]** 

```python
"crf_only": False,
"bert_path": "/home/user/yuanyh/Bert/chinese_L-12_H-768_A-12",
"init_checkpoint": "bert_model.ckpt",
"embedding": "bert"
```

关于其他参数

| 参数名 | 功能 |
| --- | --- | --- |
| normalizers | 文本预处理，现仅有数字变0处理，通过类 `Normalizers` 注册使用 |
| hidden_dim | lstm 隐层维度 |


下面参数仅 `embedding` 类模型使用

| 参数名 | 功能 |
| --- | --- |
| filter_threshold | `embedding` 模型中词库剔除低频词阈值 |
| use_seg | `embedding` 是否引入词特征 |
| token_dim | `embedding` 字向量维度 |
| seg_dim | `embedding` 词向量维度 |

### 2.3.4 flat-lattice

该版本暂未对 Flat-Lattice 模型进行封装，因此仅支持调用和预测
对应的 pipe 为 `lattice_entity_extractor`

可参考 `通用型NER配置`

重要参数包括

```python
"cache_dir": "/home/user/yuanyh/Flat_Lattice_NER/cache",
"model_dir": "/home/user/yuanyh/Flat_Lattice_NER/out"
```

### 2.3.5 bert feature

对应 pipe 为 `bert_vectors_featurizer`

重要参数包括

```python
"ip": 'localhost',
"port": 5555,
"port_out": 5556,
"http_port": 5557,
```

### 2.3.5 正则类 NER-pipe

在远古版本的 **通用型NER** 服务中，提供了部分基于正则的 NER-pipe，包括

| pipe | 功能 |
| --- | --- |
| nerIdentity | 年龄、性别、籍贯、民族 |
| nerLawAbout | 诉讼地位、法院名 |
| nerMonty | 金额 |
| nerNumber | 身份证、手机、银行卡、案件号、车牌 |


## 3.Docker服务
(待补充)

## 4.试用服务

(待补充)

## 5.未来改进及计划
(待补充)

## 6.Bug反馈及改进提交

AlexRainHao<yuanyuhaoyyh@gmail.com>
