import serial

# Configure the serial port
ser = serial.Serial('/dev/ttyACM0', 9600)  # Adjust 'COM3' to your port, and 9600 is the baud rate

def send_byte(value):
    if value == 0:
        ser.write(b'\x00')  # Send byte 0
    elif value == 1:
        ser.write(b'\x01')  # Send byte 1
    else:
        print("Invalid input, only 0 or 1 allowed.")

def main():
    while True:
        user_input = input("Enter 0 or 1 to send over serial (or 'exit' to quit): ").strip()

        if user_input == 'exit':
            print("Exiting...")
            break

        if user_input in ['0', '1']:
            send_byte(int(user_input))
        else:
            print("Please enter a valid input (0 or 1).")

if __name__ == '__main__':
    main()

# Don't forget to close the serial connection when done
ser.close()
