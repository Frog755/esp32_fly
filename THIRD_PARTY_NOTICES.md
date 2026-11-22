# Third-Party Notices

本项目自身代码以 MIT 许可证发布（见 [LICENSE](LICENSE)）。
以下文件包含或派生自第三方开源项目，其版权归原作者所有，按各自许可证使用。

## RLtools / RAPTOR

- 上游：
  - https://github.com/rl-tools/rl-tools — Copyright (c) 2023 Jonas Eschmann — MIT License
  - https://github.com/rl-tools/raptor — Copyright (c) 2025 Jonas Eschmann — MIT License
- 涉及文件：
  - `RL/data/raptor_policy.h`
  - `RL/data/rltools_gru_operations_generic.h`
  - `code/connection_test/main/raptor_policy.h`
  - `code/connection_test/main/raptor_weights.h`
  - `RL/legacy/raptor/raptor_policy_np.py`、`RL/legacy/raptor/raptor_ref.py`（参考实现移植）

```
MIT License

Copyright (c) 2023 Jonas Eschmann  (rl-tools)
Copyright (c) 2025 Jonas Eschmann  (raptor)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 参考资料（未引入代码）

- **耀坤 K1.0 硬件与飞控**：https://github.com/YaoKunS/ESP32-Drone（GPL-3.0）、oshwhub.com/lindahe/project_sulkgarf
  —— 仅在 `RL/env/yao_kun_params.py` 中引用实机物理参数，未复制其源代码。
  注意：若将本项目控制器合并进该 GPL-3.0 固件并分发，合并后的固件整体须遵守 GPL-3.0。
- **Learning to Fly in Seconds**：https://github.com/arplaboratory/learning-to-fly（MIT）—— 仅作文献与参数参考。
