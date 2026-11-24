/* bb_test.c —— 黑匣子主机单测：选槽 / 环形覆盖 / 冻结与解冻 / 模拟复位保留上一槽 / 断电随机数过滤 / 坏记录过滤，
 * 并输出一份与固件同格式的读出文本 + 期望值，由 bb_test.py 用 bb_read.py 解码对拍（C 与 Python 两套实现逐字段一致）。
 * 编译：-DBB_CAP=256 -DBB_POST=100（缩小容量，便于测环形覆盖） */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "blackbox.h"

static int ok_all = 1;
static void check(int c, const char *msg) { printf("  %s %s\n", c ? "✓" : "✗", msg); ok_all &= !!c; }

static void out_file(void *ctx, const char *line) { fprintf((FILE *)ctx, "%s\n", line); }

static bb_rec_t mk(uint32_t t, int st, int on)
{
    bb_rec_t r; memset(&r, 0, sizeof(r));
    r.t_ms = t; r.st = (uint8_t)st; r.mode = (uint8_t)(t % 2); r.flags = BB_F_LINK | BB_F_IMU_OK | (on ? BB_F_ON : 0);
    r.k = bb_u8(195.0f - (float)(t % 50)); r.q[0] = 32767; r.q[1] = bb_i16(-1234.4f); r.q[2] = (int16_t)(t % 300); r.q[3] = -5;
    r.g[0] = bb_i16((float)t * -0.5f); r.g[1] = 20000; r.g[2] = -32767; r.a[0] = -12; r.a[1] = 34; r.a[2] = 1000;
    r.z = bb_i16(480.0f); r.vz = -350; r.zt = 800; r.ab = -55; r.zb = 512; r.vb = 4150; r.thr = 1620;
    r.u[0] = 9000; r.u[1] = 10000; r.u[2] = 0; r.u[3] = 65535; r.out = on ? 2400 : 0; r.per_max = 75; r.exec_max = 67; r.vib = 123;
    return r;
}

int main(void)
{
    printf("[黑匣子] BB_CAP=%u BB_POST=%u 记录 %u 字节\n", BB_CAP, BB_POST, (unsigned)sizeof(bb_rec_t));
    bb_store_t *s = malloc(sizeof(bb_store_t));
    srand(7); for (size_t i = 0; i < sizeof(bb_store_t); i++) ((uint8_t *)s)[i] = (uint8_t)rand();   /* 断电后的随机内容 */
    bb_writer_t w;
    bb_begin(&w, s, 1);
    check(w.boot == 1 && w.cur == &s->slot[1] && !bb_slot_valid(&s->slot[0]), "断电后（随机内容）开机：boot 1 写槽 1，槽 0 作废");
    check(bb_scan(&s->slot[1]).n_valid == 0, "新槽里的随机内容全部被过滤（有效 0 条）");

    uint32_t t = 0;
    for (int i = 0; i < 600; i++) { bb_rec_t r = mk(t += 5, 0, 0); bb_write(&w, &r, 0, 0); }
    bb_range_t g = bb_scan(w.cur);
    check(g.lo == 600 - BB_CAP && g.hi == 599 && g.n_valid == BB_CAP, "环形覆盖：写 600 条后保留最后 BB_CAP 条（idx 344..599）");

    /* 一次\"飞行\"：解锁 20 拍 → 通电 120 拍 → 停转 */
    for (int i = 0; i < 20; i++) { bb_rec_t r = mk(t += 5, 1, 0); bb_write(&w, &r, 1, 0); }
    for (int i = 0; i < 120; i++) { bb_rec_t r = mk(t += 5, 2, 1); bb_write(&w, &r, 2, 1); }
    const uint32_t n_land = w.n;
    for (int i = 0; i < 400; i++) { bb_rec_t r = mk(t += 5, 0, 0); bb_write(&w, &r, 0, 0); }
    check(w.frozen && w.n == n_land + BB_POST, "停转后再记 BB_POST 条即冻结，之后的写入被丢弃");
    g = bb_scan(w.cur);
    int on_kept = 0;
    for (uint32_t p = 0; p < BB_CAP; p++) if (bb_rec_valid(w.cur, p) && (w.cur->rec[p].flags & BB_F_ON)) on_kept++;
    check(on_kept == 120 && g.hi == n_land + BB_POST - 1, "冻结后通电段 120 条完整保留");
    const uint32_t hi_frozen = g.hi;
    { bb_rec_t r = mk(t += 5, 1, 0); bb_write(&w, &r, 1, 0); }
    check(!w.frozen && bb_scan(w.cur).hi == hi_frozen + 1, "再次解锁（ARMED）→ 解冻继续记录");
    w.paused = 1; { bb_rec_t r = mk(t += 5, 1, 0); check(bb_write(&w, &r, 1, 0) == 0, "读出期间（paused）不写入"); } w.paused = 0;

    /* 写一个 \"空中复位\" 前的通电段，然后模拟复位 */
    for (int i = 0; i < 30; i++) { bb_rec_t r = mk(t += 5, 3, 1); bb_write(&w, &r, 3, 1); }
    const bb_range_t before = bb_scan(&s->slot[1]);
    bb_writer_t w2; bb_begin(&w2, s, 4 /* PANIC */);
    check(w2.boot == 2 && w2.cur == &s->slot[0] && bb_slot_valid(&s->slot[0]), "复位后开机：boot 2 改写槽 0");
    const bb_range_t after = bb_scan(&s->slot[1]);
    check(after.hi == before.hi && after.n_valid == before.n_valid && s->slot[1].rec[after.hi % BB_CAP].st == 3,
          "上一槽（出事那次）原样保留，最后一条是复位前的 RECOVER 记录");
    for (int i = 0; i < 50; i++) { bb_rec_t r = mk(t += 5, 0, 0); bb_write(&w2, &r, 0, 0); }

    /* 坏记录过滤 */
    ((uint8_t *)&s->slot[0].rec[10])[40] ^= 0x40;
    check(bb_scan(&s->slot[0]).n_valid == 49, "单字节损坏的记录被校验字节过滤");
    ((uint8_t *)&s->slot[0].rec[10])[40] ^= 0x40;

    /* 打包限幅 */
    check(bb_i16(1e9f) == 32767 && bb_i16(-1e9f) == -32767 && bb_i16(NAN) == 0 && bb_u16(-3.0f) == 0 && bb_u8(300.0f) == 255,
          "打包限幅：超量程截断、NaN → 0");

    /* 输出与固件同格式的读出文本 + 期望值（idx、t、st、几个字段），给 Python 对拍 */
    FILE *f = fopen("../build_host/bb_dump.txt", "w"), *e = fopen("../build_host/bb_expect.txt", "w");
    static uint8_t tmp[1024]; static char line[1400];
    fprintf(f, "垃圾行 F 1 2 3\nBB_BEGIN 2\n");
    uint32_t n0 = bb_dump_slot(s, 0, 1, w2.frozen, out_file, f, tmp, line);
    uint32_t n1 = bb_dump_slot(s, 1, 0, w.frozen, out_file, f, tmp, line);
    fprintf(f, "BB_DONE\n");
    for (int si = 0; si < 2; si++) {
        const bb_slot_t *h = &s->slot[si]; const bb_range_t rg = bb_scan(h);
        for (uint32_t i = rg.lo; i <= rg.hi; i++) {
            const bb_rec_t *r = &h->rec[i % BB_CAP];
            if (bb_rec_valid(h, i % BB_CAP) && r->idx == i)
                fprintf(e, "%d %u %u %u %d %d %d %u %u\n", si, r->idx, r->t_ms, r->st, r->q[1], r->g[0], r->g[2], r->u[3], r->vib);
        }
    }
    fclose(f); fclose(e);
    check(n0 == 50 && n1 == before.n_valid, "读出条数：当前槽 50、上一槽全部有效记录");

    /* 槽无效时的输出 */
    s->slot[0].magic = 0; f = fopen("../build_host/bb_dump_invalid.txt", "w");
    bb_dump_slot(s, 0, 1, 0, out_file, f, tmp, line); fclose(f);
    f = fopen("../build_host/bb_dump_invalid.txt", "r"); char buf[64] = {0}; if (!fgets(buf, sizeof buf, f)) buf[0] = 0; fclose(f);
    check(strncmp(buf, "BB_SLOT 0 invalid", 17) == 0, "槽无效时输出 BB_SLOT 0 invalid");
    free(s);
    printf(ok_all ? "黑匣子 C 单测全部通过\n" : "有失败项\n");
    return ok_all ? 0 : 1;
}
