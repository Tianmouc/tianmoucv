## 0.4.0.0起 更新日志

- 从0.4.0.0 版本起，维护更新日志
- 从0.4.0.0 版本起，增加tianmouc的仿真器模块

1. data.tianmoucData

- 对sample增加了txydiff关键字，将SD转成xy方向输出，可以用于更精确的物理保真的上采样
- datareader的其他关键词逻辑不变，以保持对旧版本兼容

1. isp.transform

- 增加了几个不同的SD空洞填补策略
- 增加了SD（矢量场数据）的旋转和镜像逻辑
- example放在tianmoucv\_example/data/rotate\_tsd.ipynb

1. sim

- 增加了简单的仿真器调用接口，支持图像序列直接转成tianmouc数据
- 输入为按照文件名排序的图像序列
  example放在tianmoucv/tianmoucv\_example/simulator

1. tianmoucv exmaple

- 对应接口更新

## 0.4.0.1 log

- txydiff关键字按照开关输出
- tianmoucv支持pip直接安装

## 0.4.0.2 log

- datareader默认改回tsdiff输出
- sd2xy接口矫正一个scaling系数
- 更新一些readme和下载资源

## 0.4.0.3 log

- cpp不再输出底层debug信息
- single读出模式的一些改善

## 0.4.0.4 log

- 支持一次性读取N帧
- data reader底层读取方法重构

## 0.4.0.5 log

- TD2VID训练，权重下载链接更新，demo更新
- pytorch load权重接口增加一个参数以支持特别新版的pytorch

## 0.4.1.0 log

- 加入数据仿真模块
- requirments严谨
- 加入相机标定模块

## 0.4.1.1 log

- open camera脚本的严重bug，和编译器相关，可能随机造成handle错误，已修
- 文档更新
- 中文文档
- optimize speed an streaming logic of ./camera module on low-end device

## 0.4.1.2 log @25.11.21

- 更新特征追踪器 tianmoucv.proc.tracking.Feature\_tracker\_sd
- 更新去模糊的新算法和权重

## 0.4.1.4 log @26.03.23

- 更新仿真器的增强逻辑
- 数据读取example的优化
- 增加nn加速的上采样

## 0.4.2.0 log @26.04.09

- TianmoucV1仿真器逻辑更新
- **开始引入AI agent**使用AI对项目进行了优化重构和文档编写

## 0.4.2.2 log @26.05.15
- 增加了一个去噪网络，并支持去噪完转存

## 0.4.2.3 log @26.05.25
- 增加了一个多路径积分SD重建算法

## 0.4.3.0 log @26.09.14

- 更新 `tianmoucv.proc.denoise`：换用当前 DDDNet 推理网络和新权重，支持自动下载与缓存、连续样本的循环状态管理、真实 TianMouC 数据预处理，以及原始/去噪 TSD 的 3×2 对比视频导出。
- 更新 `tianmoucv.proc.deblur`：换用当前 generalized STGDNet 推理网络和新权重，移除推理路径对 BasicSR、timm、einops、pandas 等额外包的依赖，并支持自动下载与缓存。
- 去模糊接口增加曝光信息读取：使用正则兼容 `us`、`µs`、`ms` 等记录形式，按照样本 `C_idx` 读取曝光时间并结合 AOP 自动估计 blur length。
- 更新 denoise/deblur 示例 notebook；denoise 使用 `outdoor_cross_6`，deblur 默认使用带 HS 标签且曝光记录完整的 `shake6` 实拍序列，并直接从 TianMouCV 公开接口完成推理，视频输出复用 `tianmoucv.utils.save_video` 的现有 MP4 写法。
- 保留 `DualRecurrentDenoiserPT`、`TianmoucDeblurNet` 和旧输入/输出键作为兼容入口；未增加 TianMouCV 之外的新依赖。

## 0.4.3.1 log @26.11.21

- 同步 `tianmoucv_dev` 的特征检测更新：Harris、Shi-Tomasi 角点检测使用张量化二阶矩与非极大值抑制；修正 HOG、多尺度 SIFT 描述子的方向、边界和坐标处理，避免修改调用方输入。
- 同步传统 LK、Horn–Schunck 光流更新：统一有符号 TD 与 SD 网格的输入约定，修正窗口求解、多尺度尺寸及位移缩放。
- 同步 `Feature_tracker_sd` 更新：使用前后向金字塔 LK 检查、周期补点和历史轨迹保留；描述子匹配使用唯一候选，并修正图像对齐的点坐标。
- 同步光流与跟踪入门、SD 特征跟踪、HS 和 LK 光流示例，更新预处理、箭头、插帧及 RGB/COP 坐标下的光流显示。
- 版本号更新为 `0.4.3.1`，Git 忽略规则增加 `codex_tmp/`；保留现有 PyPI 发布配置。

## 0.4.3.2 log @26.11.21

- SD 泊松灰度重建与 Pro GUI 对齐：在传感器原生网格上将左右 SD 转为水平、垂直梯度，用 Neumann 边界条件求解灰度；相机预览沿用 Pro 的分位数拉伸和横向二倍显示。
- RGB+SD 重建复用同一灰度解。按 COP 分辨率上采样后，以各颜色通道的局部均值校准亮度，通过曝光掩膜保留正常曝光区域，输出浮点融合图。
- 更新相机预览、USB 和 PCIe 数据读取器及重建示例。CPU 推理使用已有 OpenCV，需要自动求导或在 GPU 上运行时使用 PyTorch，无需新增 Python 依赖。
