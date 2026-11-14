# HIMLoco 步态改进：WTW 与 AMP 探索方案

目前考虑两条 HIMLoco 步态改进路线，供大家调研和实验。

| 方案 | 来源与思路 | 需要探索的内容 |
|---|---|---|
| **WTW＋HIMLoco** | 我看到重庆大学的 RC 视频后，了解到他们使用 WTW 改进 HIMLoco，实现了较好的步态。基本思路是引入步态相位、步频、支撑时间和抬脚高度等设计，引导机器人规律迈步 | 如何与 HIMLoco 结合，以及如何调整步态参数和奖励，兼顾稳定性与低速响应 |
| **AMP＋HIMLoco** | 我在 B 站看到其他使用 AMP 改进 HIMLoco 的视频，展示了较好的效果。基本思路是从参考动作数据中学习风格奖励，引导机器人形成自然、稳定的步态 | 如何获取和适配参考动作，以及如何接入 AMP、平衡动作风格与速度跟踪 |

**按目前了解到的视频展示，两条路线都有成功实践，但具体实现方式、参数和训练细节还不清楚，这是我们需要探索和验证的重点。下面的源码是研究和复现参考，不代表视频团队使用的具体版本。**

## 相关论文与源码

| 方向 | 论文 | 源码 |
|---|---|---|
| HIMLoco 基础 | [Hybrid Internal Model，ICLR 2024](https://arxiv.org/abs/2312.11460) | [官方 HIMLoco](https://github.com/InternRobotics/HIMLoco) |
| WTW | [Walk These Ways，CoRL 2022](https://arxiv.org/abs/2212.03238) | [官方 Walk These Ways](https://github.com/Improbable-AI/walk-these-ways) |
| AMP 原理 | [Adversarial Motion Priors，2021](https://arxiv.org/abs/2104.02180) | [作者项目页及代码入口](https://xbpeng.github.io/projects/AMP/index.html) |
| 四足机器人 AMP | [Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions，2022](https://arxiv.org/abs/2203.15103) | [AMP_for_hardware：A1 训练代码与参考数据](https://github.com/escontra/AMP_for_hardware) |
| HIM＋AMP 集成参考 | — | [HimLoco-AMP-Lab：Isaac Lab 实现，需自行准备专家数据](https://github.com/JF-Li5266/HimLoco-AMP-Lab)、[OpenDoge_train：包含 HIM-AMP 实现](https://github.com/OpenDogeRobotics/OpenDoge_train) |

## 最终目标与实验建议

**最终目标：形成稳定、规律的步态，能可靠响应 0.2 m/s 的速度指令；yaw 角速度跟踪更稳定，直行时减少航向漂移和左右摆动，启停与转向更加平顺。**

建议两条路线分别调研、复现，再用相同的前进、后退、直行和原地转向测试比较效果，确定可行方案。
