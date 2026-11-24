/**
 * acc_lpf.h —— 加速度计二阶低通（两级一阶级联，临界阻尼）：只喂给门控 Mahony（姿态修正 + 门控模长）
 *
 * 为什么（Stage 30 装桨振动检查实测）：装桨 40% 时振动 RMS 0.72 g，比力模长逐拍在 0.6~1.9 g 乱跳，
 *   门控只在"模长恰好落在 1 g ± 15%"的那 26% 样本上打开——这些被挑中的样本加计均值是 (−0.073, +0.056, 0.90) g，
 *   相当于倾斜 4~5°；而全部样本的均值 (0.004, 0.010, 0.989) g 和静止时一样（飞机没动、陀螺均值也只积出 0.6°）。
 *   即：误差来自门控的"选择偏差"，不是陀螺、也不是真倾斜。先低通再算模长与方向，门控看到的是平均比力，偏差消失。
 * 为什么只给 Mahony：状态机的失重（出手）/ 撞击判据、竖直滤波仍用原始加计（不能有延迟 / 不能削峰）。
 * 每级 α = dt / (τ + dt)，τ = 1 / (2π fc)；dt 按实测周期，抖动不影响截止频率。纯 C、无 ESP 依赖。
 */
#ifndef ACC_LPF_H
#define ACC_LPF_H

typedef struct { float x[3], y[3]; } acc_lpf_t;   /* x：第一级，y：输出 */

static inline void acc_lpf_init(acc_lpf_t *f, const float a0[3])
{
    for (int j = 0; j < 3; j++) { f->x[j] = a0[j]; f->y[j] = a0[j]; }
}

static inline void acc_lpf_step(acc_lpf_t *f, const float a[3], float dt, float fc)
{
    const float tau = 1.0f / (6.2831853f * fc);
    const float al = dt / (tau + dt);
    for (int j = 0; j < 3; j++) {
        f->x[j] += al * (a[j] - f->x[j]);
        f->y[j] += al * (f->x[j] - f->y[j]);
    }
}

#endif /* ACC_LPF_H */
