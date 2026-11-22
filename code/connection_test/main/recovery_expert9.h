/**
 * @file recovery_expert9.h
 * @brief YaoKun K1.0 (ESP32-S3) 微型四旋翼空中回正 —— v9 低掉高快速回正控制器（纯 C，单精度）
 *
 * 逐项对齐 RL/env/vec_expert9_env.py 的最终参数（DEFAULT_CTRL：reduced_att / prio_mixer / vbatt_aware 开启，
 * YAW_IN_FLIP 关闭），校验脚本 RL/scripts/verify_expert9_c_equivalence.py。
 *
 * 结构：
 *   1. 约简姿态测地线 bang-bang（大倾角 / 高转速）↔ 临界过阻尼 PD（近正立），迟滞 105° / 120°
 *   2. 自旋超前补偿：(τx, τy) 按 −ωz·T_DIR 预先反向旋转，抵消电机滞后期间的偏航自旋偏转
 *   3. 欧拉陀螺力矩前馈 ω×(Iω)
 *   4. 电压感知推力上限：Tmax_eff = Tmax·(Vbatt/Vref)²
 *   5. 力矩优先分配：先 roll/pitch 力矩 → 再集体推力 → 偏航最后
 *   6. 垂直通道：速度环 + 近目标抗饱和积分 + 竖直下坠刹车
 *
 * 调用约定：
 *   - 必须以 200 Hz 调用（积分步长 0.005 s、积分衰减 0.98/步、T_LAG 等均按 200 Hz 标定）
 *   - 每架飞机一个 recovery9_state_t，上电 / 抛出前调用 recovery9_reset()
 *   - q 为机体→世界姿态四元数 [w,x,y,z]（建议用 mahony_gated.h 估计），w 为机体系角速度（陀螺，rad/s）
 *   - 没有水平位置 / 速度传感器时：pos[0..1] 传 target[0..1]、vel[0..1] 传 0 —— 水平定点关闭，只做姿态回正
 *   - pos[2] 为绝对高度（m）、vel[2] 为竖直速度（m/s），z_est = 高度 − 目标高度，vz_est = 竖直速度估计
 *   - 输出 out_u[4] 为 4 路电机占空比 [0,1]，电机顺序 [FR, BR, BL, FL]（与仿真 rotor_pos 一致）
 *
 * 资源：零堆内存，单次调用 1 次 atan2f + 1 次 sinf/cosf + 若干 sqrtf，无双精度运算。
 */
#ifndef RECOVERY_EXPERT9_H
#define RECOVERY_EXPERT9_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- 机体物理常数（YaoKunK10Params） ---------------- */
#define EXP9_MASS          0.080f        /* kg */
#define EXP9_G             9.81f         /* m/s^2 */
#define EXP9_TMAX          0.27f         /* Vref 下单电机最大推力 N（= k_f·ω_max²） */
#define EXP9_VREF          3.70f         /* V */
#define EXP9_VBATT_MIN     2.80f         /* V，仿真电压下限，防除零 */
#define EXP9_ARM           0.0417193f    /* m，电机到质心的 x/y 力臂 */
#define EXP9_KM            0.012f        /* m，反扭力矩系数 */
#define EXP9_IX            5.7e-5f       /* kg·m^2 */
#define EXP9_IY            6.6e-5f
#define EXP9_IZ            1.14e-4f
/* 混控矩阵逆 Minv（T = Minv·[Fz, τx, τy, τz]）：第 0 列 = 1/4，其余列 = ±1/(4·ARM)、±1/(4·KM) */
#define EXP9_MI_XY         5.992430349f
#define EXP9_MI_Z          20.833333333f

/* ---------------- 控制参数（vec_expert9_env.DEFAULT_CTRL） ---------------- */
#define EXP9_TH_H          1.8325957f    /* 105°：低于此倾角（且转速低）退出 bang-bang */
#define EXP9_TH_HYS        0.2617994f    /* 15° 迟滞：高于 120° 进入 bang-bang */
#define EXP9_W_H           10.0f         /* |ω_xy| 迟滞阈值 rad/s（进入 = W_H+5） */
#define EXP9_T_LAG         0.09f         /* 切换曲线电机滞后超前 s */
#define EXP9_K_ALPHA       0.6f          /* 切换曲线有效角加速度比例 */
#define EXP9_R_LO          0.0f          /* r22 分界：倒扣半球 / 正立半球 */
#define EXP9_FLIP_FZ_LO    0.0f          /* 倒扣半球集体推力比例 */
#define EXP9_KD_PERP       0.002f        /* 测地线垂直方向角速度阻尼 */
#define EXP9_T_DIR         0.035f        /* 自旋超前补偿 s */
#define EXP9_K_TAU_INV     1.0f          /* 倒扣半球 bang-bang 力矩比例 */
#define EXP9_K_TAU_UP      0.6f          /* 正立半球 bang-bang 力矩比例 */
#define EXP9_YAW_RES       0.0f          /* 近正立自旋时为偏航预留的推力余量比例 */
#define EXP9_KZ_FLIP       0.002f        /* 翻转段偏航阻尼 */
#define EXP9_KP_ATT        0.025f
#define EXP9_KD_ATT        0.0027f
#define EXP9_KZ_ATT        0.002f
#define EXP9_KP_POS_XY     1.20f
#define EXP9_KD_VEL_XY     1.80f
#define EXP9_MAX_TILT_RAD  0.20f
#define EXP9_BRAKE_VZ      (-1.50f)
#define EXP9_BRAKE_TFRAC   0.95f
#define EXP9_KV_Z          9.0f
#define EXP9_KI_Z          4.0f
#define EXP9_VZ_UP         1.8f
#define EXP9_VZ_DN         (-0.8f)
#define EXP9_KZ_REF        3.0f
#define EXP9_DT            0.005f        /* 控制周期（200 Hz），仅用于高度积分 */

typedef struct {
    float   z_int;   /* 高度积分器 */
    uint8_t bb;      /* 1 = bang-bang 翻转模式，0 = PD 模式（带迟滞，跨周期保持） */
} recovery9_state_t;

static inline void recovery9_reset(recovery9_state_t *st) { st->z_int = 0.0f; st->bb = 0u; }

static inline float exp9_clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline float exp9_signf(float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); }

/* 电机 i 在 Minv 第 1/2/3 列上的符号（电机顺序 FR, BR, BL, FL） */
static const float EXP9_SGN_X[4] = {-1.0f, -1.0f,  1.0f,  1.0f};
static const float EXP9_SGN_Y[4] = {-1.0f,  1.0f,  1.0f, -1.0f};
static const float EXP9_SGN_Z[4] = {-1.0f,  1.0f, -1.0f,  1.0f};

/**
 * v9 单步（200 Hz）。返回值：本步模式（1 = bang-bang，0 = PD），便于遥测。
 */
static inline uint8_t recovery_expert9_step(
    recovery9_state_t *st,
    const float q[4], const float w[3], const float pos[3], const float vel[3],
    float vz_est, float z_est, const float target[3], float v_batt,
    float out_u[4])
{
    /* ---- 姿态 ---- */
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    {
        float n = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
        if (n > 1e-6f) { float inv = 1.0f / n; qw *= inv; qx *= inv; qy *= inv; qz *= inv; }
    }
    const float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    const float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    const float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);
    const float r01 = 2.0f * (qx * qy - qw * qz);
    const float r10 = 2.0f * (qx * qy + qw * qz);
    const float r02 = 2.0f * (qx * qz + qw * qy);
    const float r12 = 2.0f * (qy * qz - qw * qx);
    const float r20 = 2.0f * (qx * qz - qw * qy);
    const float r21 = 2.0f * (qy * qz + qw * qx);
    const float wx = w[0], wy = w[1], wz = w[2];

    /* ---- 电压感知推力上限 ---- */
    const float vb = (v_batt > EXP9_VBATT_MIN) ? v_batt : EXP9_VBATT_MIN;
    const float vr = vb / EXP9_VREF;
    const float tmax_eff = EXP9_TMAX * vr * vr;
    const float tau_rp = 2.0f * tmax_eff * EXP9_ARM;

    /* ---- 约简姿态：倾角 θ 与测地线旋转方向 n ---- */
    float nx = -r21, ny = r20;
    const float nn = sqrtf(nx * nx + ny * ny);                 /* = sin θ */
    const float theta = atan2f(nn, r22);                       /* = arccos(r22)，小角度更准 */
    const float wxy = sqrtf(wx * wx + wy * wy);
    if (nn < 0.15f) {                                          /* 近正立 / 近完全倒扣：方向退化，改用转速方向 */
        if (wxy > 0.5f) { nx = wx / wxy; ny = wy / wxy; } else { nx = 1.0f; ny = 0.0f; }
    } else {
        nx /= nn; ny /= nn;
    }

    /* ---- 测地线单轴 bang-bang ---- */
    float tau_n = tau_rp / (fabsf(nx) + fabsf(ny));            /* X 机架沿 n 的最大力矩 */
    tau_n *= (r22 < EXP9_R_LO) ? EXP9_K_TAU_INV : EXP9_K_TAU_UP;
    const float I_n = EXP9_IX * nx * nx + EXP9_IY * ny * ny;
    const float alpha_n = EXP9_K_ALPHA * tau_n / I_n;
    const float v_par = wx * nx + wy * ny;
    const float s_par = -theta + v_par * EXP9_T_LAG + v_par * fabsf(v_par) / (2.0f * alpha_n);
    const float u_par = -exp9_signf(s_par) * tau_n;
    const float wpx = wx - v_par * nx, wpy = wy - v_par * ny;
    const float tx_bb = u_par * nx - EXP9_KD_PERP * wpx;
    const float ty_bb = u_par * ny - EXP9_KD_PERP * wpy;

    /* ---- 模式迟滞 ---- */
    const int enter = (theta > EXP9_TH_H + EXP9_TH_HYS) || (wxy > EXP9_W_H + 5.0f);
    const int leave = (theta < EXP9_TH_H) && (wxy < EXP9_W_H);
    if (enter) st->bb = 1u; else if (leave) st->bb = 0u;
    const int bb = st->bb;

    /* ---- PD 段（平飞误差 + 航向解耦水平定点 + 安全门控） ---- */
    const float ex_level = r00 * r12 - r10 * r02;
    const float ey_level = r01 * r12 - r11 * r02;
    const float axw = EXP9_KP_POS_XY * (target[0] - pos[0]) - EXP9_KD_VEL_XY * vel[0];
    const float ayw = EXP9_KP_POS_XY * (target[1] - pos[1]) - EXP9_KD_VEL_XY * vel[1];
    const float abx = r00 * axw + r10 * ayw;
    const float aby = r01 * axw + r11 * ayw;
    const float d_ey = exp9_clampf(abx / EXP9_G, -EXP9_MAX_TILT_RAD, EXP9_MAX_TILT_RAD);
    const float d_ex = exp9_clampf(-aby / EXP9_G, -EXP9_MAX_TILT_RAD, EXP9_MAX_TILT_RAD);
    float beta = 0.0f;
    if (r22 > 0.65f && pos[2] > 0.8f && vel[2] > -0.5f) beta = exp9_clampf((r22 - 0.65f) / 0.25f, 0.0f, 1.0f);
    const float ex = ex_level + beta * d_ex;
    const float ey = ey_level + beta * d_ey;
    const float tx_pd = EXP9_KP_ATT * ex - EXP9_KD_ATT * wx;
    const float ty_pd = EXP9_KP_ATT * ey - EXP9_KD_ATT * wy;

    float tx = bb ? tx_bb : tx_pd;
    float ty = bb ? ty_bb : ty_pd;
    float tz = bb ? (-EXP9_KZ_FLIP * wz) : (-EXP9_KZ_ATT * wz);

    /* ---- 自旋超前补偿：(τx, τy) 旋转 φ = −ωz·T_DIR ---- */
    {
        const float phi = -wz * EXP9_T_DIR;
        const float c = cosf(phi), s = sinf(phi);
        const float tx2 = tx * c - ty * s;
        const float ty2 = tx * s + ty * c;
        tx = tx2; ty = ty2;
    }
    /* ---- 欧拉陀螺力矩前馈 ---- */
    tx += (EXP9_IZ - EXP9_IY) * wy * wz;
    ty += (EXP9_IX - EXP9_IZ) * wx * wz;
    tz += (EXP9_IY - EXP9_IX) * wx * wy;

    /* ---- 垂直通道 ---- */
    const float z_err = target[2] - pos[2];
    if (fabsf(z_err) < 0.35f && r22 > 0.75f) st->z_int = exp9_clampf(st->z_int + z_err * EXP9_DT, -0.15f, 0.15f);
    else                                     st->z_int *= 0.98f;
    const float vz_ref = exp9_clampf(-EXP9_KZ_REF * z_est, EXP9_VZ_DN, EXP9_VZ_UP);
    const float r22_lo = (r22 > 0.5f) ? r22 : 0.5f;
    const float r22_pos = (r22 > 0.0f) ? r22 : 0.0f;
    float Fz_pd = EXP9_MASS * (EXP9_G + EXP9_KV_Z * (vz_ref - vz_est) + EXP9_KI_Z * st->z_int) / r22_lo * r22_pos;
    if (Fz_pd < 0.3f * EXP9_MASS * EXP9_G) Fz_pd = 0.3f * EXP9_MASS * EXP9_G;
    const float Fmax = 4.0f * tmax_eff;
    if (!bb && vz_est < EXP9_BRAKE_VZ) Fz_pd = Fmax * EXP9_BRAKE_TFRAC;
    const float Fz_bb = (r22 > EXP9_R_LO) ? Fmax : Fmax * EXP9_FLIP_FZ_LO;
    const float Fz = bb ? Fz_bb : Fz_pd;

    /* ---- 力矩优先分配 ---- */
    float d[4], T[4];
    float dmin = 1e30f, dmax = -1e30f;
    for (int i = 0; i < 4; i++) {
        d[i] = EXP9_MI_XY * (EXP9_SGN_X[i] * tx + EXP9_SGN_Y[i] * ty);
        if (d[i] < dmin) dmin = d[i];
        if (d[i] > dmax) dmax = d[i];
    }
    float span = dmax - dmin;
    if (span < 1e-12f) span = 1e-12f;
    const float sc = (tmax_eff / span < 1.0f) ? (tmax_eff / span) : 1.0f;   /* roll/pitch 超出电机范围时整体缩放 */
    for (int i = 0; i < 4; i++) d[i] *= sc;
    dmin *= sc; dmax *= sc;
    const float c_lo = -dmin;
    float c_hi = tmax_eff - dmax;
    const float yres = (!bb && fabsf(wz) > 2.0f && r22 > 0.8f) ? EXP9_YAW_RES * tmax_eff : 0.0f;
    c_hi = (c_hi - yres > c_lo) ? (c_hi - yres) : c_lo;
    const float cc = exp9_clampf(0.25f * Fz, c_lo, c_hi);                  /* 剩余余量里放集体推力 */
    float sy = 1.0f;
    for (int i = 0; i < 4; i++) {
        T[i] = d[i] + cc;
        const float dy = EXP9_MI_Z * EXP9_SGN_Z[i] * tz;                     /* 偏航最后，按剩余余量缩放 */
        float lim = 1e30f;
        if (dy > 0.0f) lim = (tmax_eff - T[i]) / dy;
        else if (dy < 0.0f) lim = T[i] / (-dy);
        if (lim < sy) sy = lim;
    }
    if (sy < 0.0f) sy = 0.0f;
    const float inv_tmax = 1.0f / EXP9_TMAX;
    const float vcomp = EXP9_VREF / vb;
    for (int i = 0; i < 4; i++) {
        float t = T[i] + sy * EXP9_MI_Z * EXP9_SGN_Z[i] * tz;
        t = exp9_clampf(t, 0.0f, tmax_eff);
        out_u[i] = exp9_clampf(sqrtf(t * inv_tmax) * vcomp, 0.0f, 1.0f);    /* T = Tmax·(u·Vbatt/Vref)² 反解 */
    }
    return (uint8_t)bb;
}

#ifdef __cplusplus
}
#endif

#endif /* RECOVERY_EXPERT9_H */
