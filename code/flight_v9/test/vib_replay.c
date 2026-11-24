/* vib_replay.c —— 用 Stage 30 装桨振动检查的真机黑匣子数据（200 Hz）回放门控 Mahony，比较"原样"与"加计低通"两种输入
 * 飞机全程贴地不动（加计均值与通电前一致），所以估计姿态相对通电前的变化就是估计误差。
 * 输入（stdin）：每行 t gx gy gz(°/s) ax ay az(g) on
 * 编译：gcc -O2 -I../main -I../../connection_test/main vib_replay.c -lm */
#include <stdio.h>
#include <math.h>
#include "mahony_gated.h"
#include "acc_lpf.h"
#include "flight_core.h"          /* 取 FC_ACC_LPF_HZ（固件实际用的截止频率） */

#define D2R 0.01745329f
#define R2D 57.29578f
typedef struct { float t, g[3], a[3]; int on; } row_t;
static row_t R[40000];

static void euler(const float q[4], float *ro, float *pi)
{
    *ro = atan2f(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])) * R2D;
    float sp = 2 * (q[0] * q[2] - q[3] * q[1]); sp = sp > 1 ? 1 : (sp < -1 ? -1 : sp);
    *pi = asinf(sp) * R2D;
}

static float run(int n, float fc, const char *name)
{
    mahony_gated_t m; acc_lpf_t f;
    float a0[3] = {R[0].a[0] * MAHONY_G, R[0].a[1] * MAHONY_G, R[0].a[2] * MAHONY_G};
    mahony_gated_init_from_accel(&m, a0); acc_lpf_init(&f, a0);
    float r0 = 0, p0 = 0, emax = 0, esum = 0, ksum = 0; int non = 0, npre = 0, kopen = 0, first_on = -1;
    for (int i = 1; i < n; i++) {
        const float dt = R[i].t - R[i - 1].t;
        float g[3], a[3];
        for (int j = 0; j < 3; j++) { g[j] = R[i].g[j] * D2R; a[j] = R[i].a[j] * MAHONY_G; }
        const float *ain = a;
        if (fc > 0) { acc_lpf_step(&f, a, dt, fc); ain = f.y; }
        const float k = mahony_gated_update(&m, g, ain, dt);
        float ro, pi; euler(m.q, &ro, &pi);
        if (!R[i].on && first_on < 0 && R[i].t > R[0].t + 1.0f) {   /* 通电前（跳过最初 1 s 收敛） */
            r0 = (r0 * npre + ro) / (npre + 1); p0 = (p0 * npre + pi) / (npre + 1); npre++;
        }
        if (R[i].on) {
            if (first_on < 0) first_on = i;
            const float e = hypotf(ro - r0, pi - p0);
            if (e > emax) emax = e;
            esum += e; ksum += k; kopen += k > 0.5f; non++;
        }
    }
    printf("  %-24s 姿态误差 最大 %5.2f°  平均 %5.2f°   k 平均 %.2f  k>0.5 占 %3.0f%%\n", name, emax, esum / non, ksum / non, 100.0 * kopen / non);
    return emax;
}

int main(void)
{
    int n = 0;
    while (n < 40000 && scanf("%f %f %f %f %f %f %f %d", &R[n].t, &R[n].g[0], &R[n].g[1], &R[n].g[2], &R[n].a[0], &R[n].a[1], &R[n].a[2], &R[n].on) == 8) n++;
    int non = 0; for (int i = 0; i < n; i++) non += R[i].on;
    printf("回放 %d 条（通电 %d 条 = %.1f s，200 Hz）\n", n, non, non / 200.0);
    const float e_raw = run(n, 0, "原样（Stage 30 前）");
    run(n, 20, "加计低通 20 Hz");
    const float e_fw = run(n, FC_ACC_LPF_HZ, "加计低通（固件 FC_ACC_LPF_HZ）");
    run(n, 5, "加计低通 5 Hz");
    const int ok = e_raw > 5.0f && e_fw < 1.5f;
    printf("  %s 回放复现原问题（原样 > 5°）且固件低通后最大误差 < 1.5°\n", ok ? "✓" : "✗");
    return ok ? 0 : 1;
}
