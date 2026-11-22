/*
 * ESP32-S3 开发板连接与功能自检程序
 *
 * 验证三件事：
 *   1. 串口链路正常   —— 打印芯片信息（USB-Serial-JTAG）
 *   2. 程序持续运行   —— 心跳计数器递增，uptime 连续不断（未复位）
 *   3. GPIO 输出正常  —— 翻转可能的板载 LED 引脚，肉眼可见
 *
 * 编译烧录：
 *   idf.py set-target esp32s3
 *   idf.py -p /dev/ttyACM0 build flash monitor
 */

#include <stdio.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_idf_version.h"
#include "esp_mac.h"
#include "esp_heap_caps.h"

#if CONFIG_SPIRAM
#include "esp_psram.h"
#endif

static const char *TAG = "S3_SELFTEST";

/* 常见 ESP32-S3 DevKitC 板载 LED 引脚（同时翻转，哪颗亮说明对应 GPIO 正常）
 *   GPIO48 : ESP32-S3-DevKitC-1 RGB LED (WS2812)
 *   GPIO38 : 部分批次 RGB LED
 *   GPIO2  : 普通单色 LED */
#define LED_GPIO_A     48
#define LED_GPIO_B     38
#define LED_GPIO_PLAIN 2

static void print_chip_info(void)
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);

    uint32_t flash_size = 0;
    esp_flash_get_size(NULL, &flash_size);

    uint8_t mac[6] = {0};
    esp_read_mac(mac, ESP_MAC_WIFI_STA);

    printf("\n");
    printf("==========================================================\n");
    printf("       ESP32-S3 开发板自检 - 连接测试\n");
    printf("==========================================================\n");
    printf("  ESP-IDF 版本 : %s\n", esp_get_idf_version());
    printf("  目标芯片     : %s\n", CONFIG_IDF_TARGET);
    printf("  核心数       : %d\n", chip.cores);
    printf("  芯片版本     : v%d.%d\n", chip.revision / 100, chip.revision % 100);
    printf("  主频         : %d MHz\n", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    printf("  Flash 大小   : %" PRIu32 " MB\n", flash_size / (1024 * 1024));

#if CONFIG_SPIRAM
    if (esp_psram_is_initialized()) {
        printf("  PSRAM        : 已初始化, %zu MB (%zu bytes)\n",
               esp_psram_get_size() / (1024 * 1024),
               esp_psram_get_size());
    } else {
        printf("  PSRAM        : 初始化失败 ⚠️\n");
    }
#else
    printf("  PSRAM        : 未在 menuconfig 中启用\n");
#endif

#if CONFIG_SPIRAM_USE_MALLOC
    printf("  PSRAM 堆     : %" PRIu32 " bytes 可用\n",
           (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#endif

    printf("  WiFi STA MAC : %02X:%02X:%02X:%02X:%02X:%02X\n",
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    printf("  空闲堆内存   : %" PRIu32 " bytes\n", esp_get_free_heap_size());
    printf("==========================================================\n\n");
}

void app_main(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LED_GPIO_A) | (1ULL << LED_GPIO_B) | (1ULL << LED_GPIO_PLAIN),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);

    /* 等待 1.5 秒再打印自检信息：
     * USB-Serial-JTAG 在芯片启动初期还没准备好，过早打印会丢数据 */
    vTaskDelay(pdMS_TO_TICKS(1500));

    print_chip_info();
    fflush(stdout);

    ESP_LOGI(TAG, "自检程序开始运行，LED 每秒翻转一次");
    ESP_LOGI(TAG, "看到 LED 闪烁 + 下面计数器持续增长 = 开发板完全正常");

    uint32_t count = 0;
    int level = 0;

    while (1) {
        level = !level;
        gpio_set_level(LED_GPIO_A, level);
        gpio_set_level(LED_GPIO_B, level);
        gpio_set_level(LED_GPIO_PLAIN, level);

        count++;
        printf("[心跳 #%04" PRIu32 "] LED=%d  uptime=%" PRIu32 " ms  free_heap=%" PRIu32 "\n",
               count, level,
               (uint32_t)(esp_timer_get_time() / 1000),
               esp_get_free_heap_size());

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
