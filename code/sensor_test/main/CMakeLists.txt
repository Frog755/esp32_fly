idf_component_register(SRCS "sensor_test_main.c"
                       INCLUDE_DIRS "." ".." "../../connection_test/main"
                       PRIV_REQUIRES esp_driver_gpio esp_driver_spi esp_driver_i2c esp_adc esp_timer)
