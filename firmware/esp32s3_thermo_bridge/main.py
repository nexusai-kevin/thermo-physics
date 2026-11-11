"""
firmware/esp32s3_thermo_bridge/main.py
======================================
Phase 1, Step 3: MicroPython Firmware for ESP32-S3 ThermoHAL Bridge.

Runs on ESP32-S3-DevKitC-1 (MicroPython v1.23+).
Interfaces with:
  - MCP4728 4-Ch 12-bit I2C DAC (SDA=GPIO8, SCL=GPIO9)
  - 4x 12-bit SAR ADC channels (GPIO4, GPIO5, GPIO6, GPIO7) sampling the
    2N3904 avalanche noise + MCP6002 op-amp rail clipper circuit.
  - Native USB-CDC Serial (`sys.stdin.buffer` / `sys.stdout.buffer`) for binary
    `THAL` packets from `thermo_hal.py`.
"""

import struct
import sys
import time

try:
    from machine import ADC, I2C, Pin  # type: ignore
    IS_MICROPYTHON = True
except ImportError:
    IS_MICROPYTHON = False

MCP4728_ADDR = 0x60
HEADER_FMT = "<4sBBffH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class Esp32S3ThermoBridge:
    def __init__(self):
        if IS_MICROPYTHON:
            self.i2c = I2C(0, scl=Pin(9), sda=Pin(8), freq=400_000)
            self.adcs = [ADC(Pin(pin_num)) for pin_num in (4, 5, 6, 7)]
            for adc in self.adcs:
                adc.atten(ADC.ATTN_11DB)  # Full 0..3.1V range
                adc.width(ADC.WIDTH_12BIT)
        else:
            self.i2c = None
            self.adcs = []

    def write_dac_channel(self, channel: int, codeword: int, dac_bits: int) -> None:
        max_k = max(1, (1 << dac_bits) - 1)
        dac12 = min(4095, int((codeword * 4095) // max_k))
        if IS_MICROPYTHON and self.i2c is not None:
            cmd = bytearray([0x40 | ((channel & 0x03) << 1), (dac12 >> 8) & 0x0F, dac12 & 0xFF])
            try:
                self.i2c.writeto(MCP4728_ADDR, cmd)
            except OSError:
                pass

    def sample_adc_channels(self, num_channels: int) -> list[int]:
        readings = []
        for ch in range(num_channels):
            if IS_MICROPYTHON and self.adcs:
                val = self.adcs[ch & 0x03].read()
            else:
                val = 2048
            readings.append(int(val))
        return readings

    def handle_frame(self, frame_bytes: bytes) -> bytes:
        if len(frame_bytes) < HEADER_SIZE:
            return b""
        magic, dac_bits, n_ch, v_rail, noise_rms, crc16 = struct.unpack(
            HEADER_FMT, frame_bytes[:HEADER_SIZE]
        )
        if magic != b"THAL":
            return b""
        n_ch = min(16, max(1, int(n_ch)))
        payload = frame_bytes[HEADER_SIZE : HEADER_SIZE + 2 * n_ch]
        words = struct.unpack(f"<{len(payload)//2}H", payload) if len(payload) >= 2 else ()

        for ch, w in enumerate(words[:4]):
            self.write_dac_channel(ch, w, dac_bits)

        if IS_MICROPYTHON:
            time.sleep_us(50)

        adc_vals = self.sample_adc_channels(n_ch)
        resp_hdr = struct.pack("<4sHB", b"TRSP", 1, n_ch)
        resp_body = struct.pack(f"<{n_ch}H", *adc_vals)
        return resp_hdr + resp_body


def run_usb_cdc_loop() -> None:
    bridge = Esp32S3ThermoBridge()
    while True:
        hdr_bytes = sys.stdin.buffer.read(HEADER_SIZE)
        if not hdr_bytes or len(hdr_bytes) < HEADER_SIZE:
            continue
        magic, dac_bits, n_ch, v_rail, noise_rms, crc16 = struct.unpack(HEADER_FMT, hdr_bytes)
        if magic != b"THAL":
            continue
        payload = sys.stdin.buffer.read(2 * min(16, n_ch))
        resp = bridge.handle_frame(hdr_bytes + (payload or b""))
        if resp:
            sys.stdout.buffer.write(resp)
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    run_usb_cdc_loop()
