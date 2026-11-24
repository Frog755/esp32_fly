/**
 * blackbox.h —— 机载黑匣子（Stage 28）：200 Hz 定长记录写进 PSRAM 环形缓冲，复位后可读出（纯 C、无 ESP 依赖）
 *
 * 为什么：首飞必须拔 USB（电机 EMI 会让 USB 复位），串口收不到数据，要靠机上记录落地后读出。
 * 存储：bb_store_t 放在 PSRAM 的 noinit 段（EXT_RAM_NOINIT_ATTR）——开机不清零，ESP-IDF 的 PSRAM 开机自检会跳过这一段，
 *       所以软件复位 / panic / 看门狗 / USB 复位后数据还在；**断电（拔电池）会丢**。
 * 两个槽：每次开机 boot 号 +1，写 slot[boot & 1]；空中复位后新开机写另一个槽，出事那次飞行的槽保留下来。
 * 冻结：电机从"通电"变成"停转"后再记 BB_POST 条（10 s）就冻结，直到下一次解锁（ARMED）才继续写，
 *       落地后放多久都不会把这次飞行覆盖掉。读出（bbdump）期间暂停写入。
 * 不写 flash：flash 擦写会让两个核的 cache 一起停（控制循环在核 1 会停几十 ms），飞行中不能做。
 * 每条记录自带 boot 号、序号 idx（位置 = idx % 容量）和校验字节，读出时据此挑出有效记录、按 idx 排序；
 * 上电时 PSRAM 里是随机数，靠槽头 magic + 记录校验过滤掉。
 */
#ifndef BLACKBOX_H
#define BLACKBOX_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#define BB_MAGIC     0x58424259u      /* "YBBX" */
#define BB_VER       1u
#define BB_REC_SIZE  64u
#ifndef BB_CAP
#define BB_CAP       24576u           /* 每槽记录数：24576 × 5 ms = 122.9 s */
#endif
#ifndef BB_POST
#define BB_POST      2000u            /* 停转后再记 10 s 再冻结 */
#endif
#define BB_HZ        200u

/* flags 位 */
#define BB_F_LINK    0x01             /* 手机 / 串口链路在线 */
#define BB_F_IMU_OK  0x02             /* 本拍 IMU 读数非全零 */
#define BB_F_ON      0x04             /* 状态机处于通电状态（TAKEOFF / RECOVER / HOLD / LAND） */
#define BB_F_KILLED  0x08             /* 看门狗已锁死电机 */
#define BB_F_MOTOR   0x10             /* 电机驱动版固件（FLIGHT_MOTOR_OUTPUT=1） */

typedef struct __attribute__((packed)) {
    uint32_t idx;        /*  0 本槽内序号（从 0 起） */
    uint32_t t_ms;       /*  4 开机后毫秒 */
    uint16_t boot;       /*  8 开机号（与槽头一致才有效） */
    uint8_t  st, mode;   /* 10 状态机状态 0..5、控制器模式（1 = bang-bang 翻转） */
    uint8_t  ev;         /* 12 本拍新发生的事件号（fsm_event_t，0 = 无） */
    uint8_t  flags;      /* 13 BB_F_* */
    uint8_t  k;          /* 14 门控 Mahony 修正增益 ×100 */
    uint8_t  chk;        /* 15 其余 63 字节异或 ^ 0xA5 */
    int16_t  q[4];       /* 16 四元数 [w,x,y,z] ×32767 */
    int16_t  g[3];       /* 24 角速度（机体 FLU）0.1 °/s */
    int16_t  a[3];       /* 30 比力（机体 FLU）mg */
    int16_t  z, vz, zt;  /* 36 高度估计 mm、竖直速度 mm/s、目标高度 mm */
    int16_t  ab;         /* 42 竖直加计零偏估计 mm/s² */
    int16_t  zb;         /* 44 最近一次气压高度 mm */
    uint16_t vb;         /* 46 电池 mV */
    uint16_t thr;        /* 48 总推力 / 重力 ×1000 */
    uint16_t u[4];       /* 50 控制器指令 M1..M4 ×10000（限幅前） */
    uint16_t out;        /* 58 实际写入电机的占空比之和 ×10000 */
    uint8_t  per_max;    /* 60 本 5 ms 内 1 kHz 循环最长周期，单位 20 μs（255 = ≥5.1 ms） */
    uint8_t  exec_max;   /* 61 本 5 ms 内单拍最长执行时间，单位 10 μs */
    uint16_t vib;        /* 62 振动：5 个 1 kHz 样本相邻差分 |Δa| 的 RMS，mg */
} bb_rec_t;
_Static_assert(sizeof(bb_rec_t) == BB_REC_SIZE, "bb_rec_t 必须 64 字节");

typedef struct __attribute__((packed, aligned(64))) {
    uint32_t magic, ver, rec_size, cap;
    uint32_t boot, reset_reason, hchk, _r;
    uint8_t  _pad[32];
    bb_rec_t rec[BB_CAP];
} bb_slot_t;
_Static_assert(sizeof(bb_slot_t) == 64 + BB_CAP * BB_REC_SIZE, "槽头 64 字节");

typedef struct __attribute__((packed, aligned(64))) {
    uint32_t magic, boot_seq, chk, _r;
    uint8_t  _pad[48];
    bb_slot_t slot[2];
} bb_store_t;

typedef struct {
    bb_store_t *s;
    bb_slot_t *cur;
    uint32_t n;          /* 本次开机已写条数 */
    uint32_t post_left;  /* >0：停转后倒数 */
    int frozen, paused, was_on;
    uint16_t boot;
} bb_writer_t;

/* ---------------- 校验 ---------------- */
static inline uint8_t bb_rec_chk(const bb_rec_t *r)
{
    const uint8_t *p = (const uint8_t *)r; uint8_t x = 0xA5;
    for (unsigned i = 0; i < BB_REC_SIZE; i++) if (i != 15) x ^= p[i];
    return x;
}
static inline uint32_t bb_store_chk(const bb_store_t *s) { return s->magic ^ (s->boot_seq * 2654435761u) ^ 0x1234ABCDu; }
static inline uint32_t bb_slot_chk(const bb_slot_t *h)
{
    return h->magic ^ (h->ver << 8) ^ (h->rec_size << 16) ^ (h->cap * 40503u) ^ (h->boot * 2654435761u) ^ h->reset_reason ^ 0x5A5A0F0Fu;
}
static inline int bb_slot_valid(const bb_slot_t *h)
{
    return h->magic == BB_MAGIC && h->ver == BB_VER && h->rec_size == BB_REC_SIZE && h->cap == BB_CAP && h->hchk == bb_slot_chk(h);
}
static inline int bb_rec_valid(const bb_slot_t *h, uint32_t pos)
{
    const bb_rec_t *r = &h->rec[pos];
    return r->boot == (uint16_t)h->boot && r->idx % BB_CAP == pos && r->chk == bb_rec_chk(r);
}

/* ---------------- 开机：选槽 ---------------- */
static inline void bb_begin(bb_writer_t *w, bb_store_t *s, uint32_t reset_reason)
{
    uint32_t boot;
    if (s->magic == BB_MAGIC && s->chk == bb_store_chk(s)) boot = s->boot_seq + 1u;
    else {                                         /* 断电后 PSRAM 是随机数：两个槽都作废 */
        boot = 1u;
        s->slot[0].magic = 0u; s->slot[1].magic = 0u;
    }
    if ((uint16_t)boot == 0u) boot++;              /* boot 号低 16 位不为 0 */
    s->magic = BB_MAGIC; s->boot_seq = boot; s->chk = bb_store_chk(s);
    bb_slot_t *h = &s->slot[boot & 1u];
    h->magic = BB_MAGIC; h->ver = BB_VER; h->rec_size = BB_REC_SIZE; h->cap = BB_CAP;
    h->boot = boot; h->reset_reason = reset_reason; h->hchk = bb_slot_chk(h);
    memset(w, 0, sizeof(*w));
    w->s = s; w->cur = h; w->boot = (uint16_t)boot;
}

/* ---------------- 每 200 Hz 一拍：返回写入位置指针（未写返回 NULL，供调用方做 cache 回写） ----------------
 * r 的 idx / boot / chk 由这里填；st 为状态机状态，on 为本拍状态机是否通电 */
static inline bb_rec_t *bb_write(bb_writer_t *w, bb_rec_t *r, int st, int on)
{
    if (w->frozen && st == 1 /* ARMED */) { w->frozen = 0; w->post_left = 0; }    /* 新的一次飞行 */
    if (w->was_on && !on) w->post_left = BB_POST;
    if (on) w->post_left = 0;
    w->was_on = on;
    if (w->frozen || w->paused) return 0;
    r->idx = w->n; r->boot = w->boot; r->chk = bb_rec_chk(r);
    bb_rec_t *dst = &w->cur->rec[w->n % BB_CAP];
    *dst = *r;
    w->n++;
    if (w->post_left && --w->post_left == 0) w->frozen = 1;
    return dst;
}

/* ---------------- 读出：扫描一个槽，找有效记录的 idx 范围 ---------------- */
typedef struct { uint32_t lo, hi, n_valid; } bb_range_t;   /* hi 为最后一条的 idx；n_valid = 0 表示空 */
static inline bb_range_t bb_scan(const bb_slot_t *h)
{
    bb_range_t r = {0, 0, 0};
    if (!bb_slot_valid(h)) return r;
    uint32_t hi = 0; int any = 0;
    for (uint32_t p = 0; p < BB_CAP; p++)
        if (bb_rec_valid(h, p)) { r.n_valid++; if (!any || h->rec[p].idx > hi) hi = h->rec[p].idx; any = 1; }
    if (!any) return r;
    r.hi = hi; r.lo = hi + 1u > BB_CAP ? hi + 1u - BB_CAP : 0u;
    return r;
}

/* ---------------- 打包 ---------------- */
static inline int16_t bb_i16(float x)
{
    if (!(x == x)) return 0;
    if (x > 32767.0f) return 32767;
    if (x < -32767.0f) return -32767;
    return (int16_t)lrintf(x);
}
static inline uint16_t bb_u16(float x)
{
    if (!(x == x) || x < 0.0f) return 0;
    if (x > 65535.0f) return 65535;
    return (uint16_t)lrintf(x);
}
static inline uint8_t bb_u8(float x)
{
    if (!(x == x) || x < 0.0f) return 0;
    if (x > 255.0f) return 255;
    return (uint8_t)lrintf(x);
}

/* ---------------- base64 / crc32（与 Python base64 / zlib.crc32 一致） ---------------- */
static inline int bb_b64(const uint8_t *in, int n, char *out)
{
    static const char T[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int o = 0, i = 0;
    for (; i + 2 < n; i += 3) {
        const uint32_t v = (uint32_t)in[i] << 16 | (uint32_t)in[i + 1] << 8 | in[i + 2];
        out[o++] = T[v >> 18]; out[o++] = T[(v >> 12) & 63]; out[o++] = T[(v >> 6) & 63]; out[o++] = T[v & 63];
    }
    if (i < n) {
        uint32_t v = (uint32_t)in[i] << 16 | (i + 1 < n ? (uint32_t)in[i + 1] << 8 : 0u);
        out[o++] = T[v >> 18]; out[o++] = T[(v >> 12) & 63];
        out[o++] = i + 1 < n ? T[(v >> 6) & 63] : '='; out[o++] = '=';
    }
    out[o] = 0;
    return o;
}
static inline uint32_t bb_crc32(uint32_t crc, const uint8_t *p, uint32_t n)
{
    static const uint32_t T[16] = {0x00000000u, 0x1DB71064u, 0x3B6E20C8u, 0x26D930ACu, 0x76DC4190u, 0x6B6B51F4u, 0x4DB26158u, 0x5005713Cu,
                                   0xEDB88320u, 0xF00F9344u, 0xD6D6A3E8u, 0xCB61B38Cu, 0x9B64C2B0u, 0x86D3D2D4u, 0xA00AE278u, 0xBDBDF21Cu};
    crc = ~crc;
    for (uint32_t i = 0; i < n; i++) {
        crc ^= p[i];
        crc = (crc >> 4) ^ T[crc & 15u];
        crc = (crc >> 4) ^ T[crc & 15u];
    }
    return ~crc;
}

/* ---------------- 读出格式（固件串口与主机单测共用同一份代码） ----------------
 *   BB_SLOT <槽> <boot> <复位原因> <lo> <hi> <有效条数> <是否当前槽> <是否冻结>      （槽无效：BB_SLOT <槽> invalid）
 *   BB <槽> <base64：最多 16 条记录，按 idx 递增、只含有效记录>
 *   BB_END <槽> <输出条数> <输出字节的 crc32，8 位十六进制>
 * out(ctx, line) 每次输出一整行（不含换行）；tmp ≥ 1024 字节、line ≥ 1400 字节由调用方提供。返回输出条数。 */
typedef void (*bb_out_fn)(void *ctx, const char *line);
static inline int bb_u32s(char *o, uint32_t v)        /* 无 printf 依赖的十进制 */
{
    char t[11]; int n = 0, k = 0;
    do { t[n++] = (char)('0' + v % 10u); v /= 10u; } while (v);
    while (n) o[k++] = t[--n];
    o[k] = 0; return k;
}
static inline uint32_t bb_dump_slot(const bb_store_t *s, int si, int is_cur, int frozen, bb_out_fn out, void *ctx,
                                    uint8_t *tmp, char *line)
{
    const bb_slot_t *h = &s->slot[si];
    int o = 0;
    const bb_range_t rg = bb_scan(h);
    memcpy(line, "BB_SLOT ", 8); o = 8; o += bb_u32s(line + o, (uint32_t)si);
    if (!bb_slot_valid(h)) { memcpy(line + o, " invalid", 9); out(ctx, line); return 0; }
    const uint32_t f[7] = {h->boot, h->reset_reason, rg.lo, rg.hi, rg.n_valid, (uint32_t)is_cur, (uint32_t)frozen};
    for (int i = 0; i < 7; i++) { line[o++] = ' '; o += bb_u32s(line + o, f[i]); }
    out(ctx, line);
    uint32_t crc = 0, n_out = 0; int nb = 0;
    for (uint32_t idx = rg.lo; rg.n_valid && idx <= rg.hi; idx++) {
        const uint32_t p = idx % BB_CAP;
        if (bb_rec_valid(h, p) && h->rec[p].idx == idx) { memcpy(tmp + nb, &h->rec[p], BB_REC_SIZE); nb += BB_REC_SIZE; n_out++; }
        if (nb == 16 * (int)BB_REC_SIZE || (idx == rg.hi && nb)) {
            crc = bb_crc32(crc, tmp, (uint32_t)nb);
            memcpy(line, "BB ", 3); o = 3; o += bb_u32s(line + o, (uint32_t)si); line[o++] = ' ';
            bb_b64(tmp, nb, line + o);
            out(ctx, line); nb = 0;
        }
    }
    static const char H[] = "0123456789abcdef";
    memcpy(line, "BB_END ", 7); o = 7; o += bb_u32s(line + o, (uint32_t)si); line[o++] = ' ';
    o += bb_u32s(line + o, n_out); line[o++] = ' ';
    for (int i = 7; i >= 0; i--) line[o++] = H[(crc >> (4 * i)) & 15u];
    line[o] = 0;
    out(ctx, line);
    return n_out;
}

#endif /* BLACKBOX_H */
