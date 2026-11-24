idf_component_register(SRCS "flight_main.c" "web_link.c"
                       INCLUDE_DIRS "." ".." "../../connection_test/main" "../../sensor_test"
                       EMBED_TXTFILES "web/index.html"
                       PRIV_REQUIRES esp_driver_gpio esp_driver_spi esp_driver_i2c esp_driver_ledc esp_adc esp_timer
                                     esp_driver_usb_serial_jtag esp_wifi esp_netif esp_event nvs_flash esp_http_server)
# 电机输出开关：默认 0（不编译 LEDC 代码，干跑版）。
# 电机驱动版必须同时给油门上限（0~1）：idf.py -B build_mc -DFLIGHT_MOTOR_OUTPUT=1 -DFLIGHT_U_MAX=0.06 build
# 烧录电机驱动版前必须得到用户明确同意（AGENTS.md §2）；Stage 27 拆桨台架由用户运行 ./motor_bench.sh 自己烧录。
if(FLIGHT_MOTOR_OUTPUT)
    if(NOT DEFINED FLIGHT_U_MAX)
        message(FATAL_ERROR "电机驱动版必须显式给出 -DFLIGHT_U_MAX=<0~1>（例如 0.06）")
    endif()
    if(FLIGHT_U_MAX LESS_EQUAL 0 OR FLIGHT_U_MAX GREATER 1)
        message(FATAL_ERROR "FLIGHT_U_MAX=${FLIGHT_U_MAX} 超出 (0, 1]")
    endif()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FLIGHT_MOTOR_OUTPUT=1 FLIGHT_U_MAX=${FLIGHT_U_MAX}f)
endif()
