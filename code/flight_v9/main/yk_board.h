/**
 * yk_board.h —— 耀坤 K1.0（立创 ESP32-S3R8N8，PCB v2）板级驱动：电机引脚锁低、MPU6500 SPI、BMP280 I2C、电池 ADC
 * 由 code/sensor_test/main/sensor_test_main.c 的已实测驱动原样抽出（Stage 23 验证过）；只给 flight_main.c 包含一次。
 */
#pragma once
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "driver/i2c_master.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_rom_sys.h"
#include "mahony_gated.h"
#include "imu_calib.h"
#include "level_trim.h"        /* ../level_trim.h：IMU 安装水平修正 */

static const int MOTOR_PINS[4] = {16, 6, 5, 15};          /* M4 左前 / M1 右前 / M2 右后 / M3 左后 */
#define PIN_SPI_SCK   38
#define PIN_SPI_MOSI  18
#define PIN_SPI_MISO  21
#define PIN_IMU_CS    42
#define PIN_FLOW_CS   17
#define PIN_I2C_SDA   9
#define PIN_I2C_SCL   8
#define PIN_VBATT     2
#define BMP_ADDR      0x76

#define OUT(...) printf(__VA_ARGS__)

static void fatal(const char *msg) { printf("BOOT FATAL %s\n", msg); fflush(stdout); abort(); }

/* ================= 0. 电机锁定 ================= */
static void motors_lock_low(void)
{
    uint64_t mask = 0;
    for (int i = 0; i < 4; i++) mask |= 1ULL << MOTOR_PINS[i];
    gpio_config_t c = {.pin_bit_mask = mask, .mode = GPIO_MODE_INPUT_OUTPUT, .pull_up_en = GPIO_PULLUP_DISABLE,
                       .pull_down_en = GPIO_PULLDOWN_ENABLE, .intr_type = GPIO_INTR_DISABLE};
    gpio_config(&c);
    for (int i = 0; i < 4; i++) { gpio_set_level(MOTOR_PINS[i], 0); gpio_hold_en(MOTOR_PINS[i]); }
}

static void check_pin_not_motor(int pin, const char *name)
{
    for (int i = 0; i < 4; i++)
        if (pin == MOTOR_PINS[i]) { printf("BOOT FATAL pin_conflict %s=%d\n", name, pin); fatal("pin_conflict"); }
}

/* ================= 1. SPI / MPU6500 ================= */
static spi_device_handle_t s_spi_cfg, s_spi_fast;

static void imu_xfer(spi_device_handle_t dev, uint8_t reg, uint8_t *rx, int n, int write, uint8_t wval)
{
    uint8_t tx[16] = {0}, rbuf[16] = {0};
    tx[0] = write ? (reg & 0x7F) : (reg | 0x80);
    if (write) tx[1] = wval;
    spi_transaction_t t = {.length = 8 * (n + 1), .tx_buffer = tx, .rx_buffer = rbuf};
    gpio_set_level(PIN_IMU_CS, 0);
    spi_device_polling_transmit(dev, &t);
    gpio_set_level(PIN_IMU_CS, 1);
    if (rx) memcpy(rx, rbuf + 1, n);
}
static uint8_t imu_rd(uint8_t reg) { uint8_t v; imu_xfer(s_spi_cfg, reg, &v, 1, 0, 0); return v; }
static void imu_wr(uint8_t reg, uint8_t v) { imu_xfer(s_spi_cfg, reg, NULL, 1, 1, v); esp_rom_delay_us(50); }

#define ACC_LSB_PER_G   2048.0f   /* ±16 g */
#define GYR_LSB_PER_DPS 16.4f     /* ±2000 dps */

typedef struct { float a[3], g[3], temp; } imu_raw_t;

static void imu_read(imu_raw_t *o)
{
    uint8_t b[14];
    imu_xfer(s_spi_fast, 0x3B, b, 14, 0, 0);
    int16_t v[7];
    for (int i = 0; i < 7; i++) v[i] = (int16_t)((b[2 * i] << 8) | b[2 * i + 1]);
    for (int i = 0; i < 3; i++) { o->a[i] = v[i] / ACC_LSB_PER_G; o->g[i] = v[4 + i] / GYR_LSB_PER_DPS; }
    o->temp = v[3] / 333.87f + 21.0f;
}

/* 芯片原始 → 机体 FLU：加计六面校正后 ×g（m/s²），陀螺扣零偏后转 rad/s，二者同用 IMU_CHIP_TO_BODY 映射 */
static void imu_to_body(const imu_raw_t *r, const float gbias[3], float a_b[3], float g_b[3])
{
    float ac[3], gc[3];
    for (int i = 0; i < 3; i++) {
        ac[i] = (r->a[i] - IMU_ACC_OFF_G[i]) / IMU_ACC_SCALE[i] * MAHONY_G;
        gc[i] = (r->g[i] - gbias[i]) * 0.017453293f;
    }
    for (int i = 0; i < 3; i++) {
        a_b[i] = g_b[i] = 0.0f;
        for (int j = 0; j < 3; j++) { a_b[i] += IMU_CHIP_TO_BODY[i][j] * ac[j]; g_b[i] += IMU_CHIP_TO_BODY[i][j] * gc[j]; }
    }
#if LEVEL_TRIM_VALID
    {   /* IMU 安装水平修正（level_trim.h，由 level_trim.sh 测得） */
        float ta[3], tg[3];
        for (int i = 0; i < 3; i++) {
            ta[i] = LEVEL_TRIM_R[i][0] * a_b[0] + LEVEL_TRIM_R[i][1] * a_b[1] + LEVEL_TRIM_R[i][2] * a_b[2];
            tg[i] = LEVEL_TRIM_R[i][0] * g_b[0] + LEVEL_TRIM_R[i][1] * g_b[1] + LEVEL_TRIM_R[i][2] * g_b[2];
        }
        for (int i = 0; i < 3; i++) { a_b[i] = ta[i]; g_b[i] = tg[i]; }
    }
#endif
}

static int imu_init(void)
{
    gpio_config_t cs = {.pin_bit_mask = (1ULL << PIN_IMU_CS) | (1ULL << PIN_FLOW_CS), .mode = GPIO_MODE_OUTPUT};
    gpio_config(&cs);
    gpio_set_level(PIN_IMU_CS, 1);
    gpio_set_level(PIN_FLOW_CS, 1);
    spi_bus_config_t bus = {.mosi_io_num = PIN_SPI_MOSI, .miso_io_num = PIN_SPI_MISO, .sclk_io_num = PIN_SPI_SCK,
                            .quadwp_io_num = -1, .quadhd_io_num = -1, .max_transfer_sz = 32};
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_DISABLED));
    spi_device_interface_config_t d1 = {.clock_speed_hz = 1000000, .mode = 0, .spics_io_num = -1, .queue_size = 1};
    spi_device_interface_config_t d2 = {.clock_speed_hz = 8000000, .mode = 0, .spics_io_num = -1, .queue_size = 1};
    ESP_ERROR_CHECK(spi_bus_add_device(SPI2_HOST, &d1, &s_spi_cfg));
    ESP_ERROR_CHECK(spi_bus_add_device(SPI2_HOST, &d2, &s_spi_fast));
    vTaskDelay(pdMS_TO_TICKS(20));
    uint8_t who = imu_rd(0x75);
    OUT("BOOT imu_who_am_i 0x%02X\n", who);
    if (who != 0x70 && who != 0x71 && who != 0x73) return 0;
    imu_wr(0x6B, 0x80); vTaskDelay(pdMS_TO_TICKS(100));   /* 复位 */
    imu_wr(0x68, 0x07); vTaskDelay(pdMS_TO_TICKS(100));   /* 信号通路复位 */
    imu_wr(0x6B, 0x01);                                   /* PLL 时钟 */
    imu_wr(0x6A, 0x10);                                   /* I2C_IF_DIS：只走 SPI */
    imu_wr(0x19, 0x00);                                   /* 1 kHz 采样 */
    imu_wr(0x1A, 0x02);                                   /* 陀螺 DLPF 92 Hz（延迟 3.9 ms） */
    imu_wr(0x1B, 0x18);                                   /* ±2000 dps */
    imu_wr(0x1C, 0x18);                                   /* ±16 g */
    imu_wr(0x1D, 0x02);                                   /* 加计 DLPF 92 Hz */
    vTaskDelay(pdMS_TO_TICKS(50));
    uint8_t rb[5] = {imu_rd(0x1A), imu_rd(0x1B), imu_rd(0x1C), imu_rd(0x1D), imu_rd(0x6A)};
    int ok = rb[0] == 0x02 && rb[1] == 0x18 && rb[2] == 0x18 && rb[3] == 0x02 && (rb[4] & 0x10);
    OUT("BOOT imu_cfg_readback %02X %02X %02X %02X %02X %s\n", rb[0], rb[1], rb[2], rb[3], rb[4], ok ? "OK" : "MISMATCH");
    return ok;
}


/* ================= 2. I2C / BMP280 ================= */
static i2c_master_bus_handle_t s_i2c;
static i2c_master_dev_handle_t s_bmp;
static uint16_t dig_T1, dig_P1; static int16_t dig_T2, dig_T3, dig_P2, dig_P3, dig_P4, dig_P5, dig_P6, dig_P7, dig_P8, dig_P9;

static int bmp_rd(uint8_t reg, uint8_t *b, int n) { return i2c_master_transmit_receive(s_bmp, &reg, 1, b, n, 50) == ESP_OK; }
static int bmp_wr(uint8_t reg, uint8_t v) { uint8_t b[2] = {reg, v}; return i2c_master_transmit(s_bmp, b, 2, 50) == ESP_OK; }

static void i2c_scan(void)
{
    i2c_master_bus_config_t bc = {.i2c_port = -1, .sda_io_num = PIN_I2C_SDA, .scl_io_num = PIN_I2C_SCL,
                                  .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
                                  .flags.enable_internal_pullup = true};
    ESP_ERROR_CHECK(i2c_new_master_bus(&bc, &s_i2c));
    char list[256] = {0}; int n = 0, len = 0;
    for (int a = 0x08; a < 0x78; a++)
        if (i2c_master_probe(s_i2c, a, 20) == ESP_OK) { len += snprintf(list + len, sizeof(list) - len, " 0x%02X", a); n++; }
    OUT("BOOT i2c_devices %d%s\n", n, list);
}

static int bmp_init(void)
{
    if (i2c_master_probe(s_i2c, BMP_ADDR, 20) != ESP_OK) return 0;
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = BMP_ADDR, .scl_speed_hz = 400000};
    ESP_ERROR_CHECK(i2c_master_bus_add_device(s_i2c, &dc, &s_bmp));
    uint8_t id = 0; bmp_rd(0xD0, &id, 1);
    OUT("BOOT bmp_chip_id 0x%02X\n", id);
    if (id != 0x58) return 0;
    bmp_wr(0xE0, 0xB6); vTaskDelay(pdMS_TO_TICKS(10));
    uint8_t c[24];
    if (!bmp_rd(0x88, c, 24)) return 0;
#define U16(i) ((uint16_t)(c[i] | (c[i + 1] << 8)))
#define S16(i) ((int16_t)(c[i] | (c[i + 1] << 8)))
    dig_T1 = U16(0); dig_T2 = S16(2); dig_T3 = S16(4); dig_P1 = U16(6); dig_P2 = S16(8); dig_P3 = S16(10);
    dig_P4 = S16(12); dig_P5 = S16(14); dig_P6 = S16(16); dig_P7 = S16(18); dig_P8 = S16(20); dig_P9 = S16(22);
    bmp_wr(0xF5, 0x08);      /* t_sb 0.5 ms，IIR x4 */
    bmp_wr(0xF4, 0x33);      /* osrs_t x1，osrs_p x8，normal 模式 */
    vTaskDelay(pdMS_TO_TICKS(100));
    return 1;
}

/* 数据手册浮点补偿公式 */
static int bmp_read(double *p_pa, double *t_c)
{
    uint8_t b[6];
    if (!bmp_rd(0xF7, b, 6)) return 0;
    int32_t adc_P = (b[0] << 12) | (b[1] << 4) | (b[2] >> 4);
    int32_t adc_T = (b[3] << 12) | (b[4] << 4) | (b[5] >> 4);
    double v1 = (adc_T / 16384.0 - dig_T1 / 1024.0) * dig_T2;
    double v2 = (adc_T / 131072.0 - dig_T1 / 8192.0) * (adc_T / 131072.0 - dig_T1 / 8192.0) * dig_T3;
    double t_fine = v1 + v2;
    *t_c = t_fine / 5120.0;
    v1 = t_fine / 2.0 - 64000.0;
    v2 = v1 * v1 * dig_P6 / 32768.0 + v1 * dig_P5 * 2.0;
    v2 = v2 / 4.0 + dig_P4 * 65536.0;
    v1 = (dig_P3 * v1 * v1 / 524288.0 + dig_P2 * v1) / 524288.0;
    v1 = (1.0 + v1 / 32768.0) * dig_P1;
    if (v1 == 0.0) return 0;
    double p = 1048576.0 - adc_P;
    p = (p - v2 / 4096.0) * 6250.0 / v1;
    v1 = dig_P9 * p * p / 2147483648.0;
    v2 = p * dig_P8 / 32768.0;
    *p_pa = p + (v1 + v2 + dig_P7) / 16.0;
    return 1;
}

/* ================= 3. 电池电压 ================= */
static adc_oneshot_unit_handle_t s_adc; static adc_cali_handle_t s_cal; static adc_channel_t s_ch; static int s_cal_ok;

static void vbatt_init(void)
{
    adc_unit_t unit;
    ESP_ERROR_CHECK(adc_oneshot_io_to_channel(PIN_VBATT, &unit, &s_ch));
    adc_oneshot_unit_init_cfg_t uc = {.unit_id = unit};
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&uc, &s_adc));
    adc_oneshot_chan_cfg_t cc = {.atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT};
    ESP_ERROR_CHECK(adc_oneshot_config_channel(s_adc, s_ch, &cc));
    adc_cali_curve_fitting_config_t cf = {.unit_id = unit, .chan = s_ch, .atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_DEFAULT};
    s_cal_ok = adc_cali_create_scheme_curve_fitting(&cf, &s_cal) == ESP_OK;
    OUT("BOOT vbatt_adc unit=%d ch=%d cali=%d\n", (int)unit + 1, (int)s_ch, s_cal_ok);
}

static float vbatt_read(void)
{
    long sum = 0; int n = 0;
    for (int i = 0; i < 64; i++) {
        int raw, mv;
        if (adc_oneshot_read(s_adc, s_ch, &raw) != ESP_OK) continue;
        if (s_cal_ok) { adc_cali_raw_to_voltage(s_cal, raw, &mv); } else { mv = raw * 3100 / 4095; }
        sum += mv; n++;
    }
    return n ? 2.0f * (float)sum / n / 1000.0f : NAN;
}

