#include "freertos/FreeRTOS.h"
#include "esp_log.h"

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include "esp_system.h"
#include "esp_log.h"

// #include "cmd_system.h"

#include "ges_iavoz.h"

#include "command_responder.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char * TAG = "Main";





void app_main ( void ) {
    initCommandResponder();
    
    ESP_LOGI(TAG, "Starting System");
    previous_deinit_heap_size = esp_get_free_heap_size();
    ESP_LOGI(TAG, "Free heap: %d", previous_deinit_heap_size);
    
    IAVOZ_Init(1, RespondToCommand);
    previous_init_heap_size = esp_get_free_heap_size();
    ESP_LOGI(TAG, "Free heap: %d", previous_init_heap_size);
    vTaskDelay(100/portTICK_PERIOD_MS);
}

