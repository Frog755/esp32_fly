/*
 * raptor_policy.h —— Raptor 策略的纯 C 前向实现
 *
 * 网络: dense(22->16, ReLU) -> GRU(16->16, 门序 r|z|n) -> dense(16->4, Identity)
 * 目标: ESP32-S3 上尽量快的单样本前向，无动态内存分配，权重在 Flash 中
 *
 * 关键：GRU 门序和更新式严格对齐 RLtools 的 operations_generic.h，
 *      与 PyTorch 的 torch.nn.GRU 不同，不能混用。
 */

#ifndef RAPTOR_POLICY_H
#define RAPTOR_POLICY_H

#include <math.h>
#include <string.h>
#include <stdint.h>

#include "raptor_weights.h"

/* 策略状态：GRU 必须跨步维护 hidden state */
typedef struct {
    float h[RAPTOR_HIDDEN];
} raptor_state_t;

/* 复位到训练时的初始隐含状态 */
static inline void raptor_reset(raptor_state_t *st)
{
    memcpy(st->h, raptor_h0, sizeof(st->h));
}

static inline float raptor_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

/*
 * 单步前向
 *   obs : 22 维观测 [pos(3), R展平(9), lin_vel(3), ang_vel(3), prev_action(4)]
 *   act : 4 维动作输出（注意：输出层无 squash，值可能超出 [-1,1]）
 *   st  : GRU 状态，跨步保留
 */
static inline void raptor_forward(const float *obs, float *act, raptor_state_t *st)
{
    float a[RAPTOR_HIDDEN];        /* layer_0 输出 */
    float wi[RAPTOR_GRU_DIM];      /* GRU 输入分支预激活 */
    float wh[RAPTOR_GRU_DIM];      /* GRU 循环分支预激活 */
    float rz[RAPTOR_HIDDEN * 2];   /* sigmoid 后的 r,z */
    float n[RAPTOR_HIDDEN];        /* 候选状态 */

    /* ---------- layer_0: dense 22->16, ReLU ---------- */
    for (int i = 0; i < RAPTOR_HIDDEN; i++) {
        float sum = raptor_b0[i];
        const float *row = raptor_w0[i];
        for (int j = 0; j < RAPTOR_OBS_DIM; j++) {
            sum += row[j] * obs[j];
        }
        a[i] = sum > 0.0f ? sum : 0.0f;
    }

    /* ---------- layer_1: GRU 16->16 ---------- */
    /* Wi = W_in * a + b_in */
    for (int i = 0; i < RAPTOR_GRU_DIM; i++) {
        float sum = raptor_b_in[i];
        const float *row = raptor_w_in[i];
        for (int j = 0; j < RAPTOR_HIDDEN; j++) {
            sum += row[j] * a[j];
        }
        wi[i] = sum;
    }

    /* Wh = W_hid * h + b_hid */
    for (int i = 0; i < RAPTOR_GRU_DIM; i++) {
        float sum = raptor_b_hid[i];
        const float *row = raptor_w_hid[i];
        for (int j = 0; j < RAPTOR_HIDDEN; j++) {
            sum += row[j] * st->h[j];
        }
        wh[i] = sum;
    }

    /* rz = sigmoid(Wi[0:32] + Wh[0:32])，门序 [r | z] */
    for (int i = 0; i < RAPTOR_HIDDEN * 2; i++) {
        rz[i] = raptor_sigmoid(wi[i] + wh[i]);
    }

    /* n = tanh(r * Wh[32:48] + Wi[32:48])
     * 注意：r 只乘在 hidden 分支上，这是 RLtools 的写法 */
    for (int i = 0; i < RAPTOR_HIDDEN; i++) {
        float r = rz[i];
        n[i] = tanhf(r * wh[RAPTOR_HIDDEN * 2 + i] + wi[RAPTOR_HIDDEN * 2 + i]);
    }

    /* h' = z * h + (1 - z) * n */
    for (int i = 0; i < RAPTOR_HIDDEN; i++) {
        float z = rz[RAPTOR_HIDDEN + i];
        st->h[i] = z * st->h[i] + (1.0f - z) * n[i];
    }

    /* ---------- layer_2: dense 16->4, Identity ---------- */
    for (int i = 0; i < RAPTOR_ACT_DIM; i++) {
        float sum = raptor_b2[i];
        const float *row = raptor_w2[i];
        for (int j = 0; j < RAPTOR_HIDDEN; j++) {
            sum += row[j] * st->h[j];
        }
        act[i] = sum;   /* 无 squash */
    }
}

#endif /* RAPTOR_POLICY_H */
