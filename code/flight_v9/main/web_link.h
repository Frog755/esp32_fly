/**
 * web_link.h —— 手机无线控制：Wi-Fi 热点 + 网页（http://192.168.4.1）+ WebSocket
 *
 * 手机 → 板子：  "hb <seq>"（每 0.1 s 心跳）、"cmd arm|takeoff|land|disarm"
 * 板子 → 手机：  每 0.1 s 一帧遥测 JSON（由 flight_main 的 build_json 回调生成）
 * 心跳由 flight_core 计时：超过 FC_LINK_TIMEOUT（0.6 s）没心跳 = 链路丢失 → 飞行中自动降落 / 解锁中自动上锁。
 * 所有 Wi-Fi / lwIP / httpd 任务都在核 0，不占用核 1 的 1 kHz 控制。
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

#define WEB_AP_PASS "ykfly2026"
#define WEB_AP_CHANNEL 6

typedef struct {
    int  (*on_cmd)(const char *name);            /* "arm" / "takeoff" / "land" / "disarm"，认识返回 1 */
    void (*on_beat)(void);                       /* 收到一次心跳 */
    int  (*build_json)(char *buf, size_t n);     /* 生成一帧遥测，返回长度 */
} web_cb_t;

typedef struct {
    char ssid[33];
    int started, n_sta, n_ws;
    uint32_t hb_seq, hb_n;           /* 最近一次心跳序号、累计心跳数 */
    uint32_t gap_win_ms;             /* 最近 5 s 内最大心跳间隔 */
    uint32_t gap_all_ms;             /* 本次连接以来最大心跳间隔 */
    int64_t last_hb_us;
} web_stats_t;

void web_link_start(const web_cb_t *cb);
void web_link_stats(web_stats_t *out);
