/**
 * sensor_test_main.c —— 耀坤 K1.0 只读传感器实测（Stage 23）
 *
 * ★ 安全：本固件【不含任何 PWM / LEDC 代码】。上电第一件事把 4 个电机 MOSFET 栅极
 *   GPIO16 / 6 / 5 / 15 设为输出低电平并 gpio_hold_en 锁存；所有外设引脚在运行时与电机引脚比对，冲突即 abort。
 *
 * 引脚（YaoKunS/ESP32-Drone config.h 的 PCB_YaoKun_LCKFB_ESP32S3 段 + 2026-11-17 台架实测）：
 *   MPU6500  SPI  SCK 38 / MOSI 18 / MISO 21 / CS 42（PCB v2）；GPIO17 = 光流片选，保持高电平
 *   BMP280   I2C  SDA 9 / SCL 8，地址 0x76
 *   电池电压 GPIO2 ADC，100K/100K 分压（×2）
 *
 * 输出 "SEN <key> <value>" 汇总行，最后 "SEN_DONE"；之后进入 50 Hz 流式输出
 *   "S t_ms ax ay az gx gy gz"（芯片坐标系，g / dps，陀螺已扣零偏，【未校正】）供六面标定 calib.sh 使用；
 *   "A t_ms roll pitch yaw gate_k anorm_g"（机体 FLU，度；1 kHz 门控 Mahony，用 imu_calib.h 校正后的加计）供 attitude_check.py。
 * 校正参数来自 ../imu_calib.h（calib.sh 生成；重新标定后需重新编译烧录）。
 */
#include <stdio.h>
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
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_cpu.h"
#include "mahony_gated.h"
#include "imu_calib.h"       /* ../imu_calib.h：加计零偏/比例 + 芯片→机体 FLU */

/* ---------------- 引脚 ---------------- */
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

static void fatal(const char *msg) { printf("SEN FATAL %s\n", msg); fflush(stdout); abort(); }

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
        if (pin == MOTOR_PINS[i]) { printf("SEN FATAL pin_conflict %s=%d\n", name, pin); fatal("pin_conflict"); }
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
    OUT("SEN imu_who_am_i 0x%02X\n", who);
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
    OUT("SEN imu_cfg_readback %02X %02X %02X %02X %02X %s\n", rb[0], rb[1], rb[2], rb[3], rb[4], ok ? "OK" : "MISMATCH");
    return ok;
}

static void accel_range_probe(void)
{
    /* 同一静止姿态下依次切换 ±2/4/8/16 g，比较 |a|：判断误差来自量程相关的比例因子，还是固定偏置 */
    const uint8_t cfg[4] = {0x00, 0x08, 0x10, 0x18};
    const float lsb[4] = {16384.0f, 8192.0f, 4096.0f, 2048.0f};
    const int fs[4] = {2, 4, 8, 16};
    for (int r = 0; r < 4; r++) {
        imu_wr(0x1C, cfg[r]);
        vTaskDelay(pdMS_TO_TICKS(60));
        double s[3] = {0, 0, 0}, nrm = 0; const int N = 400;
        for (int k = 0; k < N; k++) {
            uint8_t b[6]; imu_xfer(s_spi_fast, 0x3B, b, 6, 0, 0);
            float a[3];
            for (int i = 0; i < 3; i++) { a[i] = (int16_t)((b[2 * i] << 8) | b[2 * i + 1]) / lsb[r]; s[i] += a[i]; }
            nrm += sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
            vTaskDelay(1);
        }
        OUT("SEN acc_range_%dg mean %.4f %.4f %.4f norm %.4f\n", fs[r], s[0] / N, s[1] / N, s[2] / N, nrm / N);
    }
    imu_wr(0x1C, 0x18);
    vTaskDelay(pdMS_TO_TICKS(60));
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
    OUT("SEN i2c_devices %d%s\n", n, list);
}

static int bmp_init(void)
{
    if (i2c_master_probe(s_i2c, BMP_ADDR, 20) != ESP_OK) return 0;
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = BMP_ADDR, .scl_speed_hz = 400000};
    ESP_ERROR_CHECK(i2c_master_bus_add_device(s_i2c, &dc, &s_bmp));
    uint8_t id = 0; bmp_rd(0xD0, &id, 1);
    OUT("SEN bmp_chip_id 0x%02X\n", id);
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
    OUT("SEN vbatt_adc unit=%d ch=%d cali=%d\n", (int)unit + 1, (int)s_ch, s_cal_ok);
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

/* ================= 4. IMU 静止统计 + 1 kHz 时序 + 门控 Mahony ================= */
typedef struct { double s[3], ss[3]; int n; } stat3_t;
static void st_add(stat3_t *s, const float v[3]) { for (int i = 0; i < 3; i++) { s->s[i] += v[i]; s->ss[i] += (double)v[i] * v[i]; } s->n++; }
static double st_mean(stat3_t *s, int i) { return s->s[i] / s->n; }
static double st_std(stat3_t *s, int i) { double m = st_mean(s, i); double v = s->ss[i] / s->n - m * m; return v > 0 ? sqrt(v) : 0; }

static float s_gbias[3];
static volatile int s_still_done;

static void still_task(void *arg)
{
    (void)arg;
    const int N = 3000;                         /* 3 s @ 1 kHz */
    stat3_t sa = {0}, sg = {0}; double anorm = 0, temp = 0;
    int64_t pmin = INT64_MAX, pmax = 0, prev = 0; int late = 0;
    uint32_t rd_max = 0; uint64_t rd_sum = 0;
    imu_raw_t r;
    TickType_t last = xTaskGetTickCount();
    for (int k = 0; k < N; k++) {
        vTaskDelayUntil(&last, 1);
        int64_t now = esp_timer_get_time();
        if (k) { int64_t p = now - prev; if (p < pmin) pmin = p; if (p > pmax) pmax = p; if (p > 1100) late++; }
        prev = now;
        uint32_t c0 = esp_cpu_get_cycle_count();
        imu_read(&r);
        uint32_t c1 = esp_cpu_get_cycle_count() - c0; if (c1 > rd_max) rd_max = c1; rd_sum += c1;
        st_add(&sa, r.a); st_add(&sg, r.g);
        anorm += sqrt(r.a[0] * r.a[0] + r.a[1] * r.a[1] + r.a[2] * r.a[2]); temp += r.temp;
    }
    for (int i = 0; i < 3; i++) s_gbias[i] = (float)st_mean(&sg, i);
    OUT("SEN imu_samples %d\n", N);
    OUT("SEN imu_period_us_min %lld\n", (long long)pmin);
    OUT("SEN imu_period_us_max %lld\n", (long long)pmax);
    OUT("SEN imu_late_gt1100us %d\n", late);
    OUT("SEN imu_read_us_avg %.1f\n", (double)rd_sum / N / 240.0);
    OUT("SEN imu_read_us_max %.1f\n", rd_max / 240.0);
    OUT("SEN acc_mean_g %.4f %.4f %.4f\n", st_mean(&sa, 0), st_mean(&sa, 1), st_mean(&sa, 2));
    OUT("SEN acc_std_mg %.2f %.2f %.2f\n", 1e3 * st_std(&sa, 0), 1e3 * st_std(&sa, 1), 1e3 * st_std(&sa, 2));
    OUT("SEN acc_norm_g %.4f\n", anorm / N);
    OUT("SEN gyro_bias_dps %.3f %.3f %.3f\n", s_gbias[0], s_gbias[1], s_gbias[2]);
    OUT("SEN gyro_std_dps %.3f %.3f %.3f\n", st_std(&sg, 0), st_std(&sg, 1), st_std(&sg, 2));
    OUT("SEN imu_temp_c %.2f\n", temp / N);

    /* 静止 5 s：纯陀螺积分漂移 + 门控 Mahony（机体 FLU，加计已按 imu_calib.h 校正） */
    float a0[3], g0[3];
    { imu_raw_t mr = {{(float)st_mean(&sa, 0), (float)st_mean(&sa, 1), (float)st_mean(&sa, 2)}, {0, 0, 0}, 0}; imu_to_body(&mr, s_gbias, a0, g0); }
    OUT("SEN calib_valid %d\n", IMU_CALIB_VALID);
    OUT("SEN acc_cal_body_g %.4f %.4f %.4f\n", a0[0] / MAHONY_G, a0[1] / MAHONY_G, a0[2] / MAHONY_G);
    OUT("SEN acc_cal_norm_g %.4f\n", sqrtf(a0[0] * a0[0] + a0[1] * a0[1] + a0[2] * a0[2]) / MAHONY_G);
    mahony_gated_t mg, mi; mahony_gated_init_from_accel(&mg, a0); mi = mg;
    double kmin = 1e9, kmax = 0, ksum = 0; int M = 5000;
    last = xTaskGetTickCount();
    for (int k = 0; k < M; k++) {
        vTaskDelayUntil(&last, 1);
        imu_read(&r);
        float g[3], a[3];
        imu_to_body(&r, s_gbias, a, g);
        float kk = mahony_gated_update(&mg, g, a, 0.001f);
        float zero[3] = {0, 0, 0}, big[3] = {0, 0, 100.0f};   /* big：比力模长远离 1g → 门控关闭 → 纯陀螺积分 */
        (void)zero;
        mahony_gated_update(&mi, g, big, 0.001f);
        if (kk < kmin) kmin = kk;
        if (kk > kmax) kmax = kk;
        ksum += kk;
    }
    /* 估计的"上"方向 vs 加速度计方向（度） */
    float qa[4]; memcpy(qa, mg.q, sizeof(qa));
    mahony_gated_t ref; mahony_gated_init_from_accel(&ref, a0);
    double dot_g = 0, dot_i = 0;
    {
        float q[4]; float v1[3], v2[3], v3[3];
        #define UP(qq, v) do { v[0] = 2.0f * (qq[1] * qq[3] - qq[0] * qq[2]); v[1] = 2.0f * (qq[2] * qq[3] + qq[0] * qq[1]); v[2] = 1.0f - 2.0f * (qq[1] * qq[1] + qq[2] * qq[2]); } while (0)
        memcpy(q, qa, sizeof(q)); UP(q, v1); UP(ref.q, v2); UP(mi.q, v3);
        dot_g = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        dot_i = v3[0] * v2[0] + v3[1] * v2[1] + v3[2] * v2[2];
    }
    OUT("SEN mahony_still_tilt_err_deg %.3f\n", acos(fmin(1.0, dot_g)) * 57.29578);
    OUT("SEN gyro_only_5s_tilt_drift_deg %.3f\n", acos(fmin(1.0, dot_i)) * 57.29578);
    OUT("SEN mahony_gate_k_min_avg_max %.3f %.3f %.3f\n", kmin, ksum / M, kmax);
    s_still_done = 1;
    vTaskDelete(NULL);
}

/* 1 kHz 姿态任务（唯一读 IMU 的任务）→ 共享快照；50 Hz 打印任务只读快照，打印阻塞不影响姿态环 */
typedef struct { imu_raw_t r; float q[4], k, anorm_g; int64_t t_us; } att_snap_t;
static att_snap_t s_snap;
static portMUX_TYPE s_snap_mux = portMUX_INITIALIZER_UNLOCKED;

static void att_task(void *arg)
{
    (void)arg;
    imu_raw_t r; float a[3], g[3];
    imu_read(&r); imu_to_body(&r, s_gbias, a, g);
    mahony_gated_t m; mahony_gated_init_from_accel(&m, a);
    int64_t prev = esp_timer_get_time();
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, 1);
        int64_t now = esp_timer_get_time();
        float dt = (now - prev) * 1e-6f; prev = now;
        if (dt > 0.005f) dt = 0.005f;
        imu_read(&r); imu_to_body(&r, s_gbias, a, g);
        float k = mahony_gated_update(&m, g, a, dt);
        float an = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / MAHONY_G;
        portENTER_CRITICAL(&s_snap_mux);
        s_snap.r = r; memcpy(s_snap.q, m.q, sizeof(m.q)); s_snap.k = k; s_snap.anorm_g = an; s_snap.t_us = now;
        portEXIT_CRITICAL(&s_snap_mux);
    }
}

static void stream_task(void *arg)
{
    (void)arg;
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, 20);     /* 50 Hz */
        att_snap_t s;
        portENTER_CRITICAL(&s_snap_mux); s = s_snap; portEXIT_CRITICAL(&s_snap_mux);
        const float w = s.q[0], x = s.q[1], y = s.q[2], z = s.q[3], R2D = 57.29578f;
        float roll = atan2f(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y)) * R2D;
        float sp = 2.0f * (w * y - z * x); sp = sp > 1.0f ? 1.0f : (sp < -1.0f ? -1.0f : sp);
        float pitch = asinf(sp) * R2D;
        float yaw = atan2f(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z)) * R2D;
        long long tms = (long long)(s.t_us / 1000);
        printf("S %lld %.4f %.4f %.4f %.3f %.3f %.3f\n", tms, s.r.a[0], s.r.a[1], s.r.a[2],
               s.r.g[0] - s_gbias[0], s.r.g[1] - s_gbias[1], s.r.g[2] - s_gbias[2]);
        printf("A %lld %.2f %.2f %.2f %.3f %.4f\n", tms, roll, pitch, yaw, s.k, s.anorm_g);
    }
}

void app_main(void)
{
    motors_lock_low();                                   /* ★ 第一件事 */
    const int used[] = {PIN_SPI_SCK, PIN_SPI_MOSI, PIN_SPI_MISO, PIN_IMU_CS, PIN_FLOW_CS, PIN_I2C_SDA, PIN_I2C_SCL, PIN_VBATT};
    const char *names[] = {"SCK", "MOSI", "MISO", "IMU_CS", "FLOW_CS", "SDA", "SCL", "VBATT"};
    for (int i = 0; i < 8; i++) check_pin_not_motor(used[i], names[i]);

    vTaskDelay(pdMS_TO_TICKS(3000));                    /* 给主机留时间打开串口 */
    OUT("SEN_BEGIN\n");
    OUT("SEN firmware sensor_test (no PWM code)\n");
    OUT("SEN reset_reason %d\n", (int)esp_reset_reason());
    OUT("SEN motor_pins_level %d %d %d %d\n", gpio_get_level(16), gpio_get_level(6), gpio_get_level(5), gpio_get_level(15));

    vbatt_init();
    float vb = vbatt_read();
    OUT("SEN vbatt_v %.3f\n", vb);
    OUT("SEN battery_connected %d\n", vb > 2.5f);

    i2c_scan();
    if (bmp_init()) {
        const int N = 150; double ps = 0, pss = 0, ts = 0, hs = 0, hss = 0; int n = 0;
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < N; i++) {
            double p, t;
            if (bmp_read(&p, &t)) {
                double h = 44330.0 * (1.0 - pow(p / 101325.0, 0.190295));
                ps += p; pss += p * p; ts += t; hs += h; hss += h * h; n++;
            }
            vTaskDelay(pdMS_TO_TICKS(20));
        }
        double dt = (esp_timer_get_time() - t0) / 1e6;
        double pm = ps / n, hm = hs / n;
        OUT("SEN bmp_samples %d in %.2f s\n", n, dt);
        OUT("SEN bmp_pressure_pa %.1f\n", pm);
        OUT("SEN bmp_temp_c %.2f\n", ts / n);
        OUT("SEN bmp_alt_std_cm %.2f\n", 100.0 * sqrt(fmax(0.0, hss / n - hm * hm)));
        OUT("SEN bmp_pressure_std_pa %.2f\n", sqrt(fmax(0.0, pss / n - pm * pm)));
    } else {
        OUT("SEN bmp_ok 0\n");
    }

    if (!imu_init()) { OUT("SEN imu_ok 0\nSEN_DONE\n"); for (;;) vTaskDelay(pdMS_TO_TICKS(1000)); }
    OUT("SEN imu_ok 1\n");
    accel_range_probe();
    xTaskCreatePinnedToCore(still_task, "still", 6144, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    while (!s_still_done) vTaskDelay(pdMS_TO_TICKS(50));
    OUT("SEN motor_pins_level_end %d %d %d %d\n", gpio_get_level(16), gpio_get_level(6), gpio_get_level(5), gpio_get_level(15));
    OUT("SEN_DONE\n");
    xTaskCreatePinnedToCore(att_task, "att", 4096, NULL, configMAX_PRIORITIES - 2, NULL, 1);
    vTaskDelay(pdMS_TO_TICKS(5));
    xTaskCreatePinnedToCore(stream_task, "stream", 4096, NULL, 5, NULL, 0);
}
