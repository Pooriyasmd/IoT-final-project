#include <ArduinoBLE.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int iteration_count = 0;
int weights_bias_cnt = 0;
int device_turn;

extern const int first_layer_input_cnt;
extern const int classes_cnt;

// two biased datasets are needed to work with our program!!!

#define DYN_NBR_WEIGHTS weights_bias_cnt
#define WEIGHTS_PER_PACKET 12

// NN parameters, set these yourself!
#define LEARNING_RATE 0.01 // The learning rate used to train your network
#define EPOCH 50          // The maximum number of epochs
#define DATA_TYPE_FLOAT   // The data type used: Set this to DATA_TYPE_DOUBLE for higher precision. However, it is better to keep this Float if you want to submit the result via BT

// You define your network in NN_def
// Right now, the network consists of three layers: 
// 1. An input layer with the size of your input as defined in the variable first_layer_input_cnt in cnn_data.h 
// 2. A hidden layer with 20 nodes
// 3. An output layer with as many classes as you defined in the variable classes_cnt in cnn_data.h 

static const unsigned int NN_def[] = {first_layer_input_cnt, 20, classes_cnt};

// Training and Validation data
#include "cnn_data_2.h"
#include "NN_functions.h" // All NN functions are stored here 

#define NBR_CENTRALS 1 

#define NBR_BATCHES_ITER (DYN_NBR_WEIGHTS / WEIGHTS_PER_PACKET)


typedef struct __attribute__( ( packed ) )
{
  int8_t device_turn;
  uint8_t batch_id;
  float w[WEIGHTS_PER_PACKET];
} bluetooth_t;

float* dyn_weights; // Allocated in main
bluetooth_t ble_data;

const char *deviceServiceUuid = "19B10000-E8F2-537E-4F6C-D104768A1214";
const char *deviceServiceInCharacteristicUuid = "19B10001-E8F2-537E-4F6C-D104768A1214";
const char *deviceServiceOutCharacteristicUuid = "19B10001-E8F2-537E-4F6C-D104768A1215";


BLEService weightsService(deviceServiceUuid);
BLECharacteristic readCharacteristic(deviceServiceInCharacteristicUuid, BLERead | BLEIndicate, sizeof(ble_data));
BLECharacteristic writeCharacteristic(deviceServiceOutCharacteristicUuid, BLEWrite, sizeof(ble_data));

void merge_new_weights() {

  for (int i = 0; i < WEIGHTS_PER_PACKET; ++i) {
   dyn_weights[ble_data.batch_id * WEIGHTS_PER_PACKET + i] = ble_data.w[i];
  }

  // merge weights
  if (ble_data.batch_id == NBR_BATCHES_ITER - 1) {
     Serial.println("Accuracy before merge:");
      printAccuracy();
      packUnpackVector(2);
      Serial.println("Accuracy after merge:");
      printAccuracy();
  }
}


void do_training() {
  Serial.print("Epoch count (training count): ");
  Serial.print(++iteration_count);
  Serial.println();

  // reordering the index for more randomness and faster learning
  shuffleIndx();
  
  // starting forward + Backward propagation
  for (int j = 0;j < numTrainData;j++) {
    generateTrainVectors(j);  
    forwardProp();
    backwardProp();
  }

  // pack the vector for bluetooth transmission
  forwardProp();
  packUnpackVector(0);
  Serial.println("Accuracy after local training:");
  printAccuracy();
}

void setup() {
  srand(0);

  Serial.begin(9600);
  delay(5000);

  // Calculate how many weights and biases we're training on the device. 
  weights_bias_cnt = calcTotalWeightsBias();

  // weights_bias_cnt has to be multiple of WEIGHTS_PER_PACKET
  int remainder = weights_bias_cnt % WEIGHTS_PER_PACKET;
  if (remainder != 0)
    weights_bias_cnt += WEIGHTS_PER_PACKET - remainder;

  Serial.print("The total number of weights and bias used for on-device training on Arduino: ");
  Serial.println(weights_bias_cnt);

  // Allocate common weight vector, and pass to setupNN, setupBLE
  DATA_TYPE* WeightBiasPtr = (DATA_TYPE*) calloc(weights_bias_cnt, sizeof(DATA_TYPE));

  setupNN(WeightBiasPtr);  // CREATES THE NETWORK BASED ON NN_def[]
  printAccuracy();

  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    setup();
  }

  dyn_weights = (float*) WeightBiasPtr;

  // set advertised local name and service UUID:
  BLE.setLocalName("Training Leader");
  BLE.setAdvertisedService(weightsService);

  weightsService.addCharacteristic(readCharacteristic);
  weightsService.addCharacteristic(writeCharacteristic);
  BLE.addService(weightsService);

  // set the initial value for the characeristic:
  writeCharacteristic.writeValue((byte *)&ble_data, sizeof(ble_data));
  readCharacteristic.writeValue((byte *)&ble_data, sizeof(ble_data));

  BLE.advertise();
  Serial.println("BLE setup complete, initial training");

  // initial train for leader
  do_training();

  
}

void loop() {
  BLE.poll();

  while(BLE.central().connected()) {
      if (writeCharacteristic.written()) {
        int8_t device_status = writeCharacteristic[0];
        
        if (device_status == -1) { 
            device_turn = 1; // reset
            ble_data.device_turn = device_turn;
          
            Serial.print("Iteration: ");
            Serial.println(iteration_count++);
          
            for (int i = 0; i < NBR_BATCHES_ITER; i++) {
              ble_data.batch_id = i;
              for (int j = 0; j < WEIGHTS_PER_PACKET; ++j) {
                ble_data.w[j] = dyn_weights[i * WEIGHTS_PER_PACKET + j];
              }
              readCharacteristic.writeValue((byte *)&ble_data, sizeof(ble_data));
          
            }
        }
    
        if (device_status == 0) {
          uint8_t batch_id = writeCharacteristic[1];
          writeCharacteristic.readValue((byte *)&ble_data, sizeof(ble_data));
          merge_new_weights();
          
          if (device_turn == NBR_CENTRALS && batch_id == NBR_BATCHES_ITER - 1) {
              device_turn = 1; // reset
              ble_data.device_turn = device_turn;
            
              Serial.print("Iteration: ");
              Serial.println(iteration_count++);
            
              for (int i = 0; i < NBR_BATCHES_ITER; i++) {
                ble_data.batch_id = i;
                for (int j = 0; j < WEIGHTS_PER_PACKET; ++j) {
                  ble_data.w[j] = dyn_weights[i * WEIGHTS_PER_PACKET + j];
                }
                readCharacteristic.writeValue((byte *)&ble_data, sizeof(ble_data));
            
              }
            do_training();
            
          } else if (ble_data.batch_id == NBR_BATCHES_ITER - 1 ) {
            ble_data.device_turn = ++device_turn;
            // just send weights
            readCharacteristic.writeValue((byte *)&ble_data, sizeof(ble_data));
          }
        }
      }


  }
}
