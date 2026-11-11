/**
 * firmware/esp32s3_thermo_bridge/esp32s3_thermo_bridge.cpp
 * ========================================================
 * Phase 1, Step 3: ESP32-S3 C++ Hardware Bridge for ThermoHAL (`thermo_hal.py`).
 *
 * Hardware Target:
 *   - MCU: ESP32-S3-DevKitC-1 (N16R8, 240 MHz dual-core Xtensa LX7, Native USB-CDC)
 *   - DAC: MCP4728 4-Channel 12-Bit I2C DAC (I2C SDA=GPIO8, SCL=GPIO9, Addr=0x60)
 *   - ADC: Internal 12-Bit SAR ADC1 (GPIO4..GPIO7) connected to MCP6002 Op-Amp
 *          + 2N3904 reverse-biased BJT avalanche/thermal noise generator.
 *
 * Wire Protocol (Binary Frame `THAL`):
 *   Request Header (16 bytes):
 *     [0..3]  magic: "THAL"
 *     [4]     dac_bits: uint8 (4, 6, 8, or 12)
 *     [5]     num_channels: uint8 (1..16)
 *     [6..9]  v_rail: float32
 *     [10..13] thermal_noise_rms: float32
 *     [14..15] crc16: uint16
 *   Payload:
 *     num_channels * uint16_t DAC codewords
 *
 *   Response Frame:
 *     "TRSP" (4 bytes) + num_samples (uint16) + num_channels (uint8) + raw 12-bit ADC uint16 array.
 */

#include <Arduino.h>
#include <Wire.h>

#define I2C_SDA_PIN 8
#define I2C_SCL_PIN 9
#define MCP4728_I2C_ADDR 0x60

static const uint8_t ADC_PINS[4] = {4, 5, 6, 7};

#pragma pack(push, 1)
struct ThalHeader {
    char magic[4];
    uint8_t dac_bits;
    uint8_t num_channels;
    float v_rail;
    float thermal_noise_rms;
    uint16_t crc16;
};
#pragma pack(pop)

// Write 12-bit value to MCP4728 channel (0..3) with software K-bit quantization mask
void writeMcp4728Channel(uint8_t channel, uint16_t codeword_kbit, uint8_t dac_bits) {
    uint16_t max_k = (1U << dac_bits) - 1U;
    if (max_k == 0) max_k = 1;
    // Scale K-bit codeword [0, 2^K - 1] to 12-bit DAC register [0, 4095]
    uint16_t dac12 = (uint16_t)(((uint32_t)codeword_kbit * 4095U) / max_k);
    if (dac12 > 4095U) dac12 = 4095U;

    Wire.beginTransmission(MCP4728_I2C_ADDR);
    // Multi-write command for single channel update: 0b01000000 | (channel << 1)
    Wire.write(0x40 | ((channel & 0x03) << 1));
    Wire.write((uint8_t)((dac12 >> 8) & 0x0F));
    Wire.write((uint8_t)(dac12 & 0xFF));
    Wire.endTransmission();
}

void setup() {
    Serial.begin(921600);
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 400000); // 400 kHz Fast-Mode I2C
    analogReadResolution(12);                     // 12-bit SAR ADC (0..4095)
    for (int i = 0; i < 4; ++i) {
        pinMode(ADC_PINS[i], INPUT);
    }
}

void loop() {
    if (Serial.available() >= (int)sizeof(ThalHeader)) {
        ThalHeader hdr;
        Serial.readBytes((char*)&hdr, sizeof(ThalHeader));

        if (memcmp(hdr.magic, "THAL", 4) != 0) {
            return;
        }

        uint8_t n_ch = hdr.num_channels > 16 ? 16 : hdr.num_channels;
        uint16_t bias_words[16] = {0};
        size_t payload_bytes = n_ch * sizeof(uint16_t);

        unsigned long t0 = millis();
        while (Serial.available() < (int)payload_bytes && (millis() - t0) < 50) {
            delayMicroseconds(50);
        }
        if (Serial.available() >= (int)payload_bytes) {
            Serial.readBytes((char*)bias_words, payload_bytes);
        }

        // 1. Drive physical MCP4728 DAC channels with K-bit quantized bias voltages
        for (uint8_t ch = 0; ch < 4 && ch < n_ch; ++ch) {
            writeMcp4728Channel(ch, bias_words[ch], hdr.dac_bits);
        }

        // Allow Op-Amp RC Ornstein-Uhlenbeck filter to settle (~50 us)
        delayMicroseconds(50);

        // 2. Sample physical 12-bit SAR ADC channels (thermal avalanche + op-amp rail clipped output)
        uint16_t adc_samples[16];
        for (uint8_t ch = 0; ch < n_ch; ++ch) {
            uint8_t pin = ADC_PINS[ch & 0x03];
            adc_samples[ch] = (uint16_t)analogRead(pin);
        }

        // 3. Return binary response frame to ThermoHAL
        Serial.write("TRSP", 4);
        uint16_t num_batches = 1;
        Serial.write((const uint8_t*)&num_batches, sizeof(uint16_t));
        Serial.write(&n_ch, sizeof(uint8_t));
        Serial.write((const uint8_t*)adc_samples, n_ch * sizeof(uint16_t));
        Serial.flush();
    }
}
