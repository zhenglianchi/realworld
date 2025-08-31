import serial
'''
设置UR5 TCP位置将夹爪末端设置为TCP
'''

class OmniPicker_Interface:
    def __init__(self,port='/dev/ttyUSB0',baudrate=2000000):
        self.port = port
        self.baudrate = baudrate
        self.set_can_baudrate = [
            0xaa,     #  0  Packet header
            0x55,     #  1  Packet header
            0x12,     #  3 Type: use variable protocol to send and receive data##  0x02- Setting (using fixed 20 byte protocol to send and receive data),   0x12- Setting (using variable protocol to send and receive data)##
            0x01,     #  3 CAN Baud Rate:  500kbps  ##  0x01(1Mbps),  0x02(800kbps),  0x03(500kbps),  0x04(400kbps),  0x05(250kbps),  0x06(200kbps),  0x07(125kbps),  0x08(100kbps),  0x09(50kbps),  0x0a(20kbps),  0x0b(10kbps),   0x0c(5kbps)##
            0x02,     #  4  Frame Type: Extended Frame  ##   0x01 standard frame,   0x02 extended frame ##
            0x00,     #  5  Filter ID1
            0x00,     #  6  Filter ID2
            0x00,     #  7  Filter ID3
            0x00,     #  8  Filter ID4
            0x00,     #  9  Mask ID1
            0x00,     #  10 Mask ID2
            0x00,     #  11 Mask ID3
            0x00,     #  12 Mask ID4
            0x00,     #  13 CAN mode:  normal mode  ##   0x00 normal mode,   0x01 silent mode,   0x02 loopback mode,   0x03 loopback silent mode ##
            0x00,     #  14 automatic resend:  automatic retransmission
            0x00,     #  15 Spare
            0x00,     #  16 Spare
            0x00,     #  17 Spare
            0x00,     #  18 Spare
        ]

    def calculate_checksum(self,data):
        checksum = sum(data[2:])
        return checksum & 0xff
    
    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate)
        # Calculate checksum
        checksum = self.calculate_checksum(self.set_can_baudrate)
        self.set_can_baudrate.append(checksum)
        #print(self.set_can_baudrate)
        self.set_can_baudrate = bytes(self.set_can_baudrate)

        # Send command to set CAN baud rate

        self.ser.write(self.set_can_baudrate)
        print(f"Connected to port {self.ser.portstr} OmniPicker")
        self.gripper_open()

    def disconnect(self):
        self.gripper_close()
        self.ser.close()
        print(f"Disconnected from port {self.ser.portstr} OmniPicker")

    def gripper_half_open(self,can_id=0x01):
        # 下行控制 位置，力度，速度，加速度，减速度
        send_can_id_data = bytes([
            0xaa,     # 0  Packet header
            0xe8,     # 1  0xc0 Tyep
            # bit5(frame type 0- standard frame (frame ID 2 bytes), 1-extended frame (frame ID 4 bytes))
            # bit4(frame format 0- data frame, 1 remote frame)
            # Bit0~3 Frame data length (0~8)
            can_id,     # 2  Frame ID data 1    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 3  Frame ID data 2    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 4  Frame ID data 3    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 5  Frame ID data 4    9~16 bit, high bytes at the front, low bytes at the back
            #数据帧
            #---------------------------------------------------------
            0x00,     # 6  Frame data 1       CAN sends  data 1
            0x7F,     # 7  Frame data 2       CAN sends  data 2
            0x7F,     # 8  Frame data 3       CAN sends  data 3
            0x7F,     # 9  Frame data 4       CAN sends  data 4
            0x7F,     # 10 Frame data 5       CAN sends  data 5
            0x7F,     # 11 Frame data 6       CAN sends  data 6
            0x00,     # 12 Frame data 7       CAN sends  data 7
            0x00,     # 13 Frame data 8       CAN sends  data 8
            #---------------------------------------------------------
            0x55,     # 14 Frame data 4       CAN sends  data 4
        ])
        self.ser.write(send_can_id_data)

    def gripper_open(self,can_id=0x01):
        # 下行控制 位置，力度，速度，加速度，减速度
        send_can_id_data = bytes([
            0xaa,     # 0  Packet header
            0xe8,     # 1  0xc0 Tyep
            # bit5(frame type 0- standard frame (frame ID 2 bytes), 1-extended frame (frame ID 4 bytes))
            # bit4(frame format 0- data frame, 1 remote frame)
            # Bit0~3 Frame data length (0~8)
            can_id,     # 2  Frame ID data 1    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 3  Frame ID data 2    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 4  Frame ID data 3    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 5  Frame ID data 4    9~16 bit, high bytes at the front, low bytes at the back
            #数据帧
            #---------------------------------------------------------
            0x00,     # 6  Frame data 1       CAN sends  data 1
            0xFF,     # 7  Frame data 2       CAN sends  data 2
            0x7F,     # 8  Frame data 3       CAN sends  data 3
            0x7F,     # 9  Frame data 4       CAN sends  data 4
            0x7F,     # 10 Frame data 5       CAN sends  data 5
            0x7F,     # 11 Frame data 6       CAN sends  data 6
            0x00,     # 12 Frame data 7       CAN sends  data 7
            0x00,     # 13 Frame data 8       CAN sends  data 8
            #---------------------------------------------------------
            0x55,     # 14 Frame data 4       CAN sends  data 4
        ])
        self.ser.write(send_can_id_data)

    def gripper_close(self,can_id=0x01):
        # 下行控制 位置，力度，速度，加速度，减速度
        send_can_id_data = bytes([
            0xaa,     # 0  Packet header
            0xe8,     # 1  0xc0 Tyep
            # bit5(frame type 0- standard frame (frame ID 2 bytes), 1-extended frame (frame ID 4 bytes))
            # bit4(frame format 0- data frame, 1 remote frame)
            # Bit0~3 Frame data length (0~8)
            can_id,     # 2  Frame ID data 1    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 3  Frame ID data 2    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 4  Frame ID data 3    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 5  Frame ID data 4    9~16 bit, high bytes at the front, low bytes at the back
            #数据帧
            #---------------------------------------------------------
            0x00,     # 6  Frame data 1       CAN sends  data 1
            0x00,     # 7  Frame data 2       CAN sends  data 2
            0x7F,     # 8  Frame data 3       CAN sends  data 3
            0x7F,     # 9  Frame data 4       CAN sends  data 4
            0x7F,     # 10 Frame data 5       CAN sends  data 5
            0x7F,     # 11 Frame data 6       CAN sends  data 6
            0x00,     # 12 Frame data 7       CAN sends  data 7
            0x00,     # 13 Frame data 8       CAN sends  data 8
            #---------------------------------------------------------
            0x55,     # 14 Frame data 4       CAN sends  data 4
        ])
        self.ser.write(send_can_id_data)

    def control(self,can_id=0x01, Pos=0.0, For=100.0, Vel=100.0, Acc=100.0, Dec=100.0):
        # 下行控制 位置，力度，速度，加速度，减速度
        Pos = int((max(0,min(100,Pos)) / 100)*255)
        For = int((max(0,min(100,For)) / 100)*255)
        Vel = int((max(0,min(100,Vel)) / 100)*255)
        Acc = int((max(0,min(100,Acc)) / 100)*255)
        Dec = int((max(0,min(100,Dec)) / 100)*255)

        send_can_id_data = bytes([
            0xaa,     # 0  Packet header
            0xe8,     # 1  0xc0 Tyep
            # bit5(frame type 0- standard frame (frame ID 2 bytes), 1-extended frame (frame ID 4 bytes))
            # bit4(frame format 0- data frame, 1 remote frame)
            # Bit0~3 Frame data length (0~8)
            can_id,     # 2  Frame ID data 1    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 3  Frame ID data 2    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 4  Frame ID data 3    1~8 bit, high bytes at the front, low bytes at the back
            0x00,     # 5  Frame ID data 4    9~16 bit, high bytes at the front, low bytes at the back
            #数据帧
            #---------------------------------------------------------
            0x00,     # 6  Frame data 1       CAN sends  data 1
            Pos,     # 7  Frame data 2       CAN sends  data 2
            For,     # 8  Frame data 3       CAN sends  data 3
            Vel,     # 9  Frame data 4       CAN sends  data 4
            Acc,     # 10 Frame data 5       CAN sends  data 5
            Dec,     # 11 Frame data 6       CAN sends  data 6
            0x00,     # 12 Frame data 7       CAN sends  data 7
            0x00,     # 13 Frame data 8       CAN sends  data 8
            #---------------------------------------------------------
            0x55,     # 14 Frame data 4       CAN sends  data 4
        ])
        self.ser.write(send_can_id_data)


