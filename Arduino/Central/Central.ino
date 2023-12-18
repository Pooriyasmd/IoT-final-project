#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <ArduinoBLE.h>

int iter_cnt = 0;
int weights_bias_cnt = 0;
extern const int first_layer_input_cnt;
extern const int classes_cnt;

// two biased datasets are needed to work

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
#include "cnn_data_1.h"  
#include "NN_functions.h" // All NN functions are stored here 
#define CENTRAL_ID 1

#define NBR_BATCHES_ITER (DYN_NBR_WEIGHTS / WEIGHTS_PER_PACKET)



typedef struct __attribute__( ( packed ) )
{
  int8_t device_turn;
  uint8_t batch_id;
  float w[WEIGHTS_PER_PACKET];
} bluetooth_t;

float* dyn_weights;
bluetooth_t ble_data;
BLEDevice peripheral;

BLECharacteristic readCharacteristic;
BLECharacteristic writeCharacteristic;
const char *deviceServiceInCharacteristicUuid = "19B10001-E8F2-537E-4F6C-D104768A1214";
const char *deviceServiceOutCharacteristicUuid = "19B10001-E8F2-537E-4F6C-D104768A1215";



void connectPeripheral() {
    Serial.println("Attempting connection ...");
    
    if (peripheral.connect()) {
      Serial.println("Connection successful");
    } else {
      Serial.println("Connection failed!");
      return;
    }
    
    Serial.println("Scanning for attributes ...");
    if (peripheral.discoverAttributes()) {
      Serial.println("Attributes found");
    } else {
      Serial.println("Attribute discovery unsuccessful!");
      peripheral.disconnect();
      return;
    }

  readCharacteristic = peripheral.characteristic(deviceServiceInCharacteristicUuid);
  writeCharacteristic = peripheral.characteristic(deviceServiceOutCharacteristicUuid);

  if (!readCharacteristic.subscribe()) {
    Serial.println("Subscription failed!");
    peripheral.disconnect();
    return;
  }

  // connection established
  ble_data.device_turn = -1;
  writeCharacteristic.writeValue((byte *)&ble_data, sizeof(ble_data));

  loopPeripheral();

  Serial.println("Peripheral disconnected");
}

void do_training() {
  packUnpackVector(1);
  Serial.println("Accuracy using incoming weights:");
  printAccuracy();


  Serial.print("Epoch count (training count): ");
  Serial.print(++iter_cnt);
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

void loopPeripheral() {
  while (peripheral.connected()) {
    if (readCharacteristic.valueUpdated()) {

      readCharacteristic.readValue((byte *)&ble_data, sizeof(ble_data));

      if (ble_data.device_turn == 1) {
        //for (int i = 0; i < WEIGHTS_PER_PACKET; ++i) {
          //  dyn_weights[ble_data.batch_id * WEIGHTS_PER_PACKET + i] = ble_data.w[i];
        //}
        memcpy(dyn_weights + ble_data.batch_id * WEIGHTS_PER_PACKET, ble_data.w, WEIGHTS_PER_PACKET * sizeof(ble_data.w[0]));
        if (ble_data.batch_id == NBR_BATCHES_ITER - 1) {
         
          do_training();
        }
      }

      if (ble_data.device_turn == CENTRAL_ID && (ble_data.batch_id == NBR_BATCHES_ITER - 1 || CENTRAL_ID != 1)) {
        ble_data.device_turn = 0;

        for (int i = 0; i < NBR_BATCHES_ITER; i++) {
          ble_data.batch_id = i;
          // Copy weights
          //for (int j = 0; j < WEIGHTS_PER_PACKET; ++j) {
           //   ble_data.w[j] = dyn_weights[i * WEIGHTS_PER_PACKET + j];
         // }

          memcpy(ble_data.w, dyn_weights + i * WEIGHTS_PER_PACKET, WEIGHTS_PER_PACKET * sizeof(ble_data.w[0]));

          writeCharacteristic.writeValue((byte *)&ble_data, sizeof(ble_data));
        }
      }
    }
  }
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

  dyn_weights = (float*) WeightBiasPtr;    // we only support float for BLE transmission
  // initialize the BLE hardware
  BLE.begin();

  Serial.println("BLE starting scanning for peripheral");
  BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");

}

void loop() {
    peripheral = BLE.available();

  if (peripheral) {
    Serial.print("**  Found ");
    Serial.print(peripheral.address());
    Serial.print(" '");
    Serial.print(peripheral.localName());
    Serial.print("' ");
    Serial.print(peripheral.advertisedServiceUuid());
    Serial.println();

    if (peripheral.localName() != "Training Leader") {
      return;
    }

    BLE.stopScan();

    connectPeripheral();

    BLE.scanForUuid("19b10000-e8f2-537e-4f6c-d104768a1214");

    if (iter_cnt >= EPOCH) {
      Serial.println("Finished training, shutting down.");
      printAccuracy();
      BLE.stopAdvertise();
      BLE.disconnect();
    }
  }
}
