#include "model_handling.h"


ErrorReporter* error_reporter = NULL;
MicroMutableOpResolver* micro_op_resolver = NULL;
ModelSettings_t * ms = NULL;

bool model_init(IAVoz_System_t ** sysptr, IAVoz_ModelSettings_t * ms, pIAVOZCallback_t cb) {
    int previous_heap_size = esp_get_free_heap_size();
    IAVoz_System_t *sys = (IAVoz_System_t * ) malloc(sizeof(IAVoz_System_t));
    ESP_LOGI(TAG, "Heap difference for sys: %ld", esp_get_free_heap_size() - previous_heap_size);
    
    ESP_LOGI(TAG, "Creating error reporter");
    error_reporter = tflite::GetMicroErrorReporter();  // GetMicroErrorReporter obtains a pointer to a singleton, it is not necessary to delete after it is done.
    if (!sys->error_reporter) {
        ESP_LOGE(TAG, "Could not obtain error reporter"); 
        return false;
    }

    ESP_LOGI(TAG, "Adding operations to op resolver");
    micro_op_resolver = new tflite::MicroMutableOpResolver<9>(error_reporter);
    if (!add_model_layers(sys->micro_op_resolver)) {
        ESP_LOGE(TAG, "Could not add layers to op resolver"); 
        return false;
    }

    ESP_LOGI(TAG, "Initializing model");
    if (!init_model_descriptor(&(sys->command_model), g_command_model, g_command_model_len, sys)) {
        ESP_LOGE(TAG, "Error while initializing model"); 
        return false;
    }  
    
    if ( !cb ) {
        ESP_LOGE(TAG, "Null callback provided"); 
        return false;
    }
        
    ESP_LOGI(TAG, "Initializing RecognizeCommands");
    recognizer = new RecognizeCommands(error_reporter);

    ESP_LOGI(TAG, "Model Init finished");

    return true;

}

bool init_model_descriptor(IAVOZ_model_t * model_descriptor, const unsigned char * model, size_t model_len, IAVoz_System_t * sys) {
    model_descriptor->tensor_arena = (uint8_t *) malloc(model_len);

    // TF API
    model_descriptor->model = tflite::GetModel(model);
    if (model_descriptor->model->version() != TFLITE_SCHEMA_VERSION){
        ESP_LOGE(TAG, "Model provided is schema version %ld not equal to supported version %d.", model_descriptor->model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    ESP_LOGI(TAG, "Creating micro interpreter");
    model_descriptor->interpreter = new tflite::MicroInterpreter(model_descriptor->model, *(sys->micro_op_resolver), model_descriptor->tensor_arena, model_len, sys->error_reporter);

    ESP_LOGI(TAG, "Allocating tensors for model");
    TfLiteStatus allocate_status = model_descriptor->interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return false;
    }

    // Get information about the memory area to use for the model's input.
    model_descriptor->input = model_descriptor->interpreter->input(0);
    if ((model_descriptor->input->dims->size != 2) || (model_descriptor->input->dims->data[0] != 1) || 
        (model_descriptor->input->dims->data[1] != (sys->ms->kFeatureSliceCount * sys->ms->kFeatureSliceSize)) || 
        (model_descriptor->input->type != kTfLiteInt8)) 
    {
        ESP_LOGE(TAG, "Bad input tensor parameters");
        return false;
    }

    model_descriptor->input_buffer = model_descriptor->input->data.int8;

    return true;
}


bool add_model_layers(tflite::MicroMutableOpResolver<9> * op_resolver) {
    if (op_resolver->AddDepthwiseConv2D() != kTfLiteOk) {ESP_LOGE(TAG, "Could not add deppth wise conv 2D layer");  return false;}
    if (op_resolver->AddFullyConnected() != kTfLiteOk)  {ESP_LOGE(TAG, "Could not add fully connected layer");      return false;}
    if (op_resolver->AddSoftmax() != kTfLiteOk)         {ESP_LOGE(TAG, "Could not add softmax layer");              return false;}
    if (op_resolver->AddReshape() != kTfLiteOk)         {ESP_LOGE(TAG, "Could not add reshape layer");              return false;}
    if (op_resolver->AddMaxPool2D() != kTfLiteOk)       {ESP_LOGE(TAG, "Could not add max pool 2D layer");          return false;}
    if (op_resolver->AddConv2D() != kTfLiteOk)          {ESP_LOGE(TAG, "Could not add conv 2D layer");              return false;}
    if (op_resolver->AddPad() != kTfLiteOk)             {ESP_LOGE(TAG, "Could not add Pad layer");                  return false;}
    if (op_resolver->AddAdd() != kTfLiteOk)             {ESP_LOGE(TAG, "Could not add Add layer");                  return false;}
    if (op_resolver->AddMean() != kTfLiteOk)            {ESP_LOGE(TAG, "Could not add Mean layer");                 return false;}

    return true;
}




