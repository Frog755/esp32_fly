idf_component_register(SRCS "flight_main.c" "web_link.c"
                       INCLUDE_DIRS "." ".." "../../connection_test/main" "../../sensor_test"
                       EMBED_TXTFILES "web/index.html"
                       PRIV_REQUIRES esp_driver_gpio esp_driver_spi esp_driver_i2c esp_driver_ledc esp_adc esp_timer
                                     esp_driver_usb_serial_jtag esp_wifi esp_netif esp_event nvs_flash esp_http_server)
# 电机输出开关：默认 0（不编译 LEDC 代码）。仅做编译检查：idf.py -B build_mc -DFLIGHT_MOTOR_OUTPUT=1 build（不要烧录）
if(FLIGHT_MOTOR_OUTPUT)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FLIGHT_MOTOR_OUTPUT=1)
endif()
