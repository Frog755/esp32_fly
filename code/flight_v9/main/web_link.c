/* web_link.c —— Wi-Fi 热点 + HTTP + WebSocket（见 web_link.h） */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_http_server.h"
#include "web_link.h"

extern const char index_html_start[] asm("_binary_index_html_start");
extern const char index_html_end[] asm("_binary_index_html_end");

static httpd_handle_t s_srv;
static web_cb_t s_cb;
static web_stats_t s_st;
static portMUX_TYPE s_wmux = portMUX_INITIALIZER_UNLOCKED;
static volatile int s_send_busy;

/* ---------------- Wi-Fi 热点 ---------------- */
static void wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg; (void)base; (void)data;
    if (id == WIFI_EVENT_AP_STACONNECTED) { s_st.n_sta++; printf("WEB phone_connected n=%d\n", s_st.n_sta); }
    else if (id == WIFI_EVENT_AP_STADISCONNECTED) { if (s_st.n_sta > 0) s_st.n_sta--; printf("WEB phone_disconnected n=%d\n", s_st.n_sta); }
}

static void wifi_ap_start(void)
{
    esp_err_t r = nvs_flash_init();
    if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND) { ESP_ERROR_CHECK(nvs_flash_erase()); r = nvs_flash_init(); }
    ESP_ERROR_CHECK(r);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t ic = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&ic));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL));
    uint8_t mac[6]; esp_read_mac(mac, ESP_MAC_WIFI_SOFTAP);
    snprintf(s_st.ssid, sizeof(s_st.ssid), "YK-FLY-%02X%02X", mac[4], mac[5]);
    wifi_config_t wc = {0};
    memcpy(wc.ap.ssid, s_st.ssid, strlen(s_st.ssid)); wc.ap.ssid_len = (uint8_t)strlen(s_st.ssid);
    memcpy(wc.ap.password, WEB_AP_PASS, strlen(WEB_AP_PASS));
    wc.ap.channel = WEB_AP_CHANNEL; wc.ap.max_connection = 2; wc.ap.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_ps(WIFI_PS_NONE);
}

/* ---------------- HTTP / WebSocket ---------------- */
static esp_err_t root_get(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, index_html_start, index_html_end - index_html_start - 1);   /* EMBED_TXTFILES 末尾带 \0 */
}

static void handle_text(const char *t)
{
    if (!strncmp(t, "hb ", 3)) {
        const int64_t now = esp_timer_get_time();
        portENTER_CRITICAL(&s_wmux);
        if (s_st.last_hb_us) {
            const uint32_t gap = (uint32_t)((now - s_st.last_hb_us) / 1000);
            if (gap > s_st.gap_win_ms) s_st.gap_win_ms = gap;
            if (gap > s_st.gap_all_ms && gap < 60000) s_st.gap_all_ms = gap;
        }
        s_st.last_hb_us = now; s_st.hb_seq = (uint32_t)strtoul(t + 3, NULL, 10); s_st.hb_n++;
        portEXIT_CRITICAL(&s_wmux);
        s_cb.on_beat();
    } else if (!strncmp(t, "cmd ", 4)) {
        const char *c = t + 4;
        if (s_cb.on_cmd(c)) printf("WEB cmd %s\n", c);
    }
}

static esp_err_t ws_handler(httpd_req_t *req)
{
    if (req->method == HTTP_GET) {
        printf("WEB websocket_open fd=%d\n", httpd_req_to_sockfd(req));
        portENTER_CRITICAL(&s_wmux); s_st.last_hb_us = 0; s_st.gap_all_ms = 0; portEXIT_CRITICAL(&s_wmux);
        return ESP_OK;
    }
    httpd_ws_frame_t f; memset(&f, 0, sizeof(f));
    esp_err_t r = httpd_ws_recv_frame(req, &f, 0);
    if (r != ESP_OK) return r;
    if (f.len == 0) return ESP_OK;
    if (f.type != HTTPD_WS_TYPE_TEXT || f.len > 60) return ESP_FAIL;      /* 网页只发短文本；其他一律断开 */
    char buf[64]; f.payload = (uint8_t *)buf;
    r = httpd_ws_recv_frame(req, &f, f.len);
    if (r != ESP_OK) return r;
    buf[f.len] = 0;
    handle_text(buf);
    return ESP_OK;
}

typedef struct { char *txt; size_t len; } send_job_t;

static void send_all(void *arg)
{
    send_job_t *j = (send_job_t *)arg;
    size_t n = 8; int fds[8], nws = 0;
    if (httpd_get_client_list(s_srv, &n, fds) == ESP_OK) {
        for (size_t i = 0; i < n; i++) {
            if (httpd_ws_get_fd_info(s_srv, fds[i]) != HTTPD_WS_CLIENT_WEBSOCKET) continue;
            httpd_ws_frame_t f = {.type = HTTPD_WS_TYPE_TEXT, .payload = (uint8_t *)j->txt, .len = j->len, .final = true};
            if (httpd_ws_send_frame_async(s_srv, fds[i], &f) == ESP_OK) nws++;
        }
    }
    s_st.n_ws = nws;
    free(j->txt); free(j);
    s_send_busy = 0;
}

static void telem_web_task(void *arg)
{
    (void)arg;
    TickType_t last = xTaskGetTickCount();
    int k = 0;
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(100));
        if (++k % 50 == 0) { portENTER_CRITICAL(&s_wmux); s_st.gap_win_ms = 0; portEXIT_CRITICAL(&s_wmux); }   /* 5 s 窗口 */
        if (s_send_busy || !s_srv) continue;
        send_job_t *j = malloc(sizeof(*j));
        char *t = malloc(1024);
        if (!j || !t) { free(j); free(t); continue; }
        int len = s_cb.build_json(t, 1024);
        if (len <= 0 || len >= 1024) { free(j); free(t); continue; }
        j->txt = t; j->len = (size_t)len;
        s_send_busy = 1;
        if (httpd_queue_work(s_srv, send_all, j) != ESP_OK) { free(t); free(j); s_send_busy = 0; }
    }
}

void web_link_start(const web_cb_t *cb)
{
    s_cb = *cb;
    wifi_ap_start();
    httpd_config_t hc = HTTPD_DEFAULT_CONFIG();
    hc.core_id = 0; hc.max_open_sockets = 4; hc.lru_purge_enable = true; hc.stack_size = 6144;
    ESP_ERROR_CHECK(httpd_start(&s_srv, &hc));
    static const httpd_uri_t root = {.uri = "/", .method = HTTP_GET, .handler = root_get};
    static const httpd_uri_t ws = {.uri = "/ws", .method = HTTP_GET, .handler = ws_handler, .is_websocket = true};
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_srv, &root));
    ESP_ERROR_CHECK(httpd_register_uri_handler(s_srv, &ws));
    xTaskCreatePinnedToCore(telem_web_task, "telem_web", 4096, NULL, 4, NULL, 0);
    s_st.started = 1;
    printf("BOOT wifi_ap ssid %s pass %s url http://192.168.4.1/\n", s_st.ssid, WEB_AP_PASS);
}

void web_link_stats(web_stats_t *out)
{
    portENTER_CRITICAL(&s_wmux); *out = s_st; portEXIT_CRITICAL(&s_wmux);
}
