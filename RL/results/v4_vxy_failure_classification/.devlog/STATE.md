# Current Project State

- **Active Goal**: 完成 v4_vxy 标准评估并交付失败分类结论。
- **Current Phase**: 本轮评估与判读已完成。
- **Recent Completed**:
  - 固定种子四档难度各 100 回合，成功率为 45% / 32% / 25% / 22%。
  - 已完成失败分类与 v3_reward 同条件对比。
- **Next Priorities**:
  - [ ] 拆分 `out_of_bounds` 的具体根因。
  - [ ] 针对中低难度边界逃逸、极限难度未翻正分别设计下一轮训练假设。
- **Blockers / Known Risks**: 失败分类仍受当前 `out_of_bounds` 合并判据限制；下一轮需增加更细终止原因记录。
